`timescale 1ns / 1ps
`include "npu_params.vh"

// DeepSeek V3 Multi-Head Latent Attention (MLA) Core
// Compresses KV Cache into a latent vector $c_t$ (Down Projection)
// and decompresses it back into Keys and Values (Up Projection)

module mla_core_v3 (
    input wire clk,
    input wire rst_n,
    input wire [63:0] hidden_state_in, // 4-dim vector (Q8.8)
    
    output reg [31:0] latent_c_out,    // 2-dim latent vector (Q8.8)
    output reg [63:0] key_out,         // 4-dim key vector (Q8.8)
    output reg [63:0] value_out,       // 4-dim value vector (Q8.8)
    output reg out_valid
);

    // Unpack 4-dim Input
    wire signed [15:0] h [0:3];
    assign h[0] = hidden_state_in[15:0];
    assign h[1] = hidden_state_in[31:16];
    assign h[2] = hidden_state_in[47:32];
    assign h[3] = hidden_state_in[63:48];

    // --- Hardcoded Projection Matrices ---
    // Down Proj (4x2 matrix): W_DKV
    wire signed [15:0] W_down [0:1][0:3];
    assign W_down[0][0] = 16'h0080; assign W_down[0][1] = 16'h0080; assign W_down[0][2] = 16'h0000; assign W_down[0][3] = 16'h0000;
    assign W_down[1][0] = 16'h0000; assign W_down[1][1] = 16'h0000; assign W_down[1][2] = 16'h0080; assign W_down[1][3] = 16'h0080;

    // Up Proj Keys (2x4 matrix): W_UK
    wire signed [15:0] W_up_k [0:3][0:1];
    assign W_up_k[0][0] = 16'h0100; assign W_up_k[0][1] = 16'h0000;
    assign W_up_k[1][0] = 16'h0100; assign W_up_k[1][1] = 16'h0000;
    assign W_up_k[2][0] = 16'h0000; assign W_up_k[2][1] = 16'h0100;
    assign W_up_k[3][0] = 16'h0000; assign W_up_k[3][1] = 16'h0100;

    // Up Proj Values (2x4 matrix): W_UV
    wire signed [15:0] W_up_v [0:3][0:1];
    assign W_up_v[0][0] = -16'h0100; assign W_up_v[0][1] = 16'h0000;
    assign W_up_v[1][0] = -16'h0100; assign W_up_v[1][1] = 16'h0000;
    assign W_up_v[2][0] = 16'h0000;  assign W_up_v[2][1] = -16'h0100;
    assign W_up_v[3][0] = 16'h0000;  assign W_up_v[3][1] = -16'h0100;

    // --- Pipeline Stage 1: Down Projection ---
    reg signed [31:0] c_t_raw [0:1]; // Accumulators
    integer i, j;
    always @(posedge clk) begin
        if (!rst_n) begin
            c_t_raw[0] <= 0;
            c_t_raw[1] <= 0;
        end else begin
            c_t_raw[0] <= (h[0] * W_down[0][0]) + (h[1] * W_down[0][1]) + (h[2] * W_down[0][2]) + (h[3] * W_down[0][3]);
            c_t_raw[1] <= (h[0] * W_down[1][0]) + (h[1] * W_down[1][1]) + (h[2] * W_down[1][2]) + (h[3] * W_down[1][3]);
        end
    end

    // Shift to Q8.8 format
    wire signed [15:0] c_t [0:1];
    assign c_t[0] = c_t_raw[0][23:8];
    assign c_t[1] = c_t_raw[1][23:8];

    // --- Pipeline Stage 2: Up Projection ---
    reg signed [31:0] k_t_raw [0:3];
    reg signed [31:0] v_t_raw [0:3];
    
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i=0; i<4; i=i+1) k_t_raw[i] <= 0;
            for (i=0; i<4; i=i+1) v_t_raw[i] <= 0;
            latent_c_out <= 0;
            out_valid <= 0;
        end else begin
            for (i=0; i<4; i=i+1) begin
                k_t_raw[i] <= (c_t[0] * W_up_k[i][0]) + (c_t[1] * W_up_k[i][1]);
                v_t_raw[i] <= (c_t[0] * W_up_v[i][0]) + (c_t[1] * W_up_v[i][1]);
            end
            latent_c_out <= {c_t[1], c_t[0]};
            out_valid <= 1'b1;
        end
    end

    // --- Output Assignment ---
    always @(*) begin
        key_out[15:0]  = k_t_raw[0][23:8];
        key_out[31:16] = k_t_raw[1][23:8];
        key_out[47:32] = k_t_raw[2][23:8];
        key_out[63:48] = k_t_raw[3][23:8];
        
        value_out[15:0]  = v_t_raw[0][23:8];
        value_out[31:16] = v_t_raw[1][23:8];
        value_out[47:32] = v_t_raw[2][23:8];
        value_out[63:48] = v_t_raw[3][23:8];
    end

endmodule
