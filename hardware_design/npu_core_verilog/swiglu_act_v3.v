`timescale 1ns / 1ps
`include "npu_params.vh"

// DeepSeek V3 SwiGLU Activation Hardware Unit
// Implements: Out = (x * HardSigmoid(x)) * y
// HardSigmoid(x) = max(0, min(1, (x + 3) / 6))
// Assuming inputs x and y are in a signed Q16.16 format for precision.

module swiglu_act_v3 (
    input wire clk,
    input wire rst_n,
    input wire signed [31:0] x, // Gate input (e.g. from down_proj)
    input wire signed [31:0] y, // Data input (e.g. from up_proj)
    
    output reg signed [31:0] out,
    output reg out_valid
);

    // Q16.16 format constants
    localparam signed [31:0] CONST_3 = 32'h0003_0000;
    localparam signed [31:0] CONST_6 = 32'h0006_0000;
    // 1/6 in Q16.16 is approx 0.166666... => 65536 / 6 = 10922
    localparam signed [31:0] CONST_INV_6 = 32'h0000_2AAA; 

    // Pipeline Stage 1: Add 3, Clamp, Align Data
    reg signed [31:0] x_plus_3;
    reg signed [31:0] x_p1;
    reg signed [31:0] y_p1;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            x_plus_3 <= 0;
            x_p1 <= 0;
            y_p1 <= 0;
        end else begin
            x_p1 <= x;
            y_p1 <= y;
            if (x < -CONST_3) begin
                x_plus_3 <= 0;
            end else if (x > CONST_3) begin
                x_plus_3 <= CONST_6;
            end else begin
                x_plus_3 <= x + CONST_3;
            end
        end
    end

    // Pipeline Stage 2: HardSigmoid = (x_plus_3) * (1/6)
    // Mult Q16.16 * Q16.16 = Q32.32
    reg signed [31:0] hard_sig;
    reg signed [31:0] x_p2;
    reg signed [31:0] y_p2;
    wire signed [63:0] sig_mult = x_plus_3 * CONST_INV_6;

    always @(posedge clk) begin
        if (!rst_n) begin
            hard_sig <= 0;
            x_p2 <= 0;
            y_p2 <= 0;
        end else begin
            x_p2 <= x_p1;
            y_p2 <= y_p1;
            hard_sig <= sig_mult[47:16]; // Extract Q16.16
        end
    end

    // Pipeline Stage 3: Swish(x) = x * HardSigmoid
    reg signed [31:0] swish_x;
    reg signed [31:0] y_p3;
    wire signed [63:0] swish_mult = x_p2 * hard_sig;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            swish_x <= 0;
            y_p3 <= 0;
        end else begin
            swish_x <= swish_mult[47:16];
            y_p3 <= y_p2;
        end
    end

    // Pipeline Stage 4: Out = Swish(x) * y
    wire signed [63:0] final_mult = swish_x * y_p3;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            out <= 0;
            out_valid <= 0;
        end else begin
            out <= final_mult[47:16];
            out_valid <= 1'b1;
        end
    end

endmodule
