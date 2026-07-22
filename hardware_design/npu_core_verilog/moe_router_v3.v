`timescale 1ns / 1ps
`include "npu_params.vh"

// DeepSeek V3 Mixture-of-Experts (MoE) Router
// Computes token routing by taking dot product of input vector
// against expert centroids and selecting the top expert (Max Logit).

module moe_router_v3 #(
    parameter NUM_EXPERTS = 4,
    parameter VEC_SIZE = 4
)(
    input wire clk,
    input wire rst_n,
    // Flattened input token vector (4 elements of 16-bit Q8.8)
    input wire [63:0] token_vec_in,
    
    // Expert selection output
    output reg [1:0] selected_expert,
    output reg out_valid
);

    // Hardcoded expert centroids for demonstration (in a real ASIC, these are SRAM)
    // 4 Experts, each has 4 features (Q8.8 format)
    wire signed [15:0] experts [0:3][0:3];
    
    // Expert 0: [1.0, 0.0, 0.5, -0.5]
    assign experts[0][0] = 16'h0100; assign experts[0][1] = 16'h0000;
    assign experts[0][2] = 16'h0080; assign experts[0][3] = -16'h0080;
    // Expert 1: [0.0, 1.0, -0.5, 0.5]
    assign experts[1][0] = 16'h0000; assign experts[1][1] = 16'h0100;
    assign experts[1][2] = -16'h0080; assign experts[1][3] = 16'h0080;
    // Expert 2: [-1.0, 0.0, 1.0, 0.0]
    assign experts[2][0] = -16'h0100; assign experts[2][1] = 16'h0000;
    assign experts[2][2] = 16'h0100; assign experts[2][3] = 16'h0000;
    // Expert 3: [0.0, -1.0, 0.0, 1.0]
    assign experts[3][0] = 16'h0000; assign experts[3][1] = -16'h0100;
    assign experts[3][2] = 16'h0000; assign experts[3][3] = 16'h0100;

    // Unpack token vector
    wire signed [15:0] token [0:3];
    assign token[0] = token_vec_in[15:0];
    assign token[1] = token_vec_in[31:16];
    assign token[2] = token_vec_in[47:32];
    assign token[3] = token_vec_in[63:48];

    // Pipeline Stage 1: Parallel Dot Products
    reg signed [31:0] logits [0:3];
    integer i, j;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < 4; i = i + 1) logits[i] <= 0;
        end else begin
            for (i = 0; i < 4; i = i + 1) begin
                logits[i] <= (token[0] * experts[i][0]) + 
                             (token[1] * experts[i][1]) + 
                             (token[2] * experts[i][2]) + 
                             (token[3] * experts[i][3]);
            end
        end
    end

    // Pipeline Stage 2: Comparator Tree (Top-1 Selection)
    reg [1:0] win_01, win_23;
    reg signed [31:0] val_01, val_23;

    always @(posedge clk) begin
        if (!rst_n) begin
            win_01 <= 0; val_01 <= 0;
            win_23 <= 0; val_23 <= 0;
        end else begin
            if (logits[0] > logits[1]) begin win_01 <= 2'd0; val_01 <= logits[0]; end
            else                       begin win_01 <= 2'd1; val_01 <= logits[1]; end
            
            if (logits[2] > logits[3]) begin win_23 <= 2'd2; val_23 <= logits[2]; end
            else                       begin win_23 <= 2'd3; val_23 <= logits[3]; end
        end
    end

    // Pipeline Stage 3: Final Winner
    always @(posedge clk) begin
        if (!rst_n) begin
            selected_expert <= 0;
            out_valid <= 0;
        end else begin
            if (val_01 > val_23) selected_expert <= win_01;
            else                 selected_expert <= win_23;
            out_valid <= 1'b1;
        end
    end

endmodule
