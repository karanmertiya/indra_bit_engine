`timescale 1ns / 1ps
`include "llm_params.vh"

module rmsnorm (
    input wire [`EMBED_DIM*`DATA_W-1:0] in_vec,
    output reg [`EMBED_DIM*`DATA_W-1:0] out_vec
);

    reg signed [`DATA_W-1:0] X [0:`EMBED_DIM-1];
    reg [31:0] sum_sq;
    reg [15:0] rms_sq;
    reg [15:0] inv_rms; // 1 / sqrt(rms_sq)
    
    // Static weights for RMSNorm (gamma) - Simulated ROM
    reg signed [`DATA_W-1:0] gamma [0:`EMBED_DIM-1];
    integer g;
    initial begin
        for (g = 0; g < `EMBED_DIM; g = g + 1) begin
            gamma[g] = 16'h0100; // Initialize gamma to 1.0 (Q8.8)
        end
    end

    integer i;
    always @(*) begin
        // 1. Unpack input and calculate sum of squares
        sum_sq = 0;
        for (i = 0; i < `EMBED_DIM; i = i + 1) begin
            X[i] = in_vec[i*`DATA_W +: `DATA_W];
            // Multiply x * x, shift right to keep Q8.8
            sum_sq = sum_sq + ((X[i] * X[i]) >> `FRAC_W);
        end
        
        // 2. Mean of squares: divide by EMBED_DIM (which is 32, so >> 5)
        rms_sq = sum_sq >> 5; 
        // Adding epsilon (e.g., 1 in Q8.8 is 0x0001)
        if (rms_sq == 0) rms_sq = 1;
        
        // 3. Inverse Square Root LUT (approximate for Q8.8)
        // In real hardware, we use Newton-Raphson or a large LUT.
        // Here we just use a simplified piecewise mapping.
        if (rms_sq < 16'h0010)      inv_rms = 16'h0400; // Very small rms, large inv (4.0)
        else if (rms_sq < 16'h0040) inv_rms = 16'h0200; // 2.0
        else if (rms_sq < 16'h0100) inv_rms = 16'h0155; // 1.33
        else if (rms_sq < 16'h0400) inv_rms = 16'h0080; // 0.5
        else if (rms_sq < 16'h1000) inv_rms = 16'h0040; // 0.25
        else                        inv_rms = 16'h0020; // 0.125
        
        // 4. Multiply inputs by inv_rms and gamma
        for (i = 0; i < `EMBED_DIM; i = i + 1) begin
            automatic reg signed [31:0] norm_x = (X[i] * inv_rms) >>> `FRAC_W;
            automatic reg signed [31:0] scaled_x = (norm_x * gamma[i]) >>> `FRAC_W;
            out_vec[i*`DATA_W +: `DATA_W] = scaled_x;
        end
    end

endmodule
