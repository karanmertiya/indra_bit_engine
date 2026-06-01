`timescale 1ns / 1ps
`include "npu_params.vh"

module rope_rotate (
    input wire clk,
    input wire signed [`INT32_W-1:0] x0, // Even dimension element
    input wire signed [`INT32_W-1:0] x1, // Odd dimension element
    input wire [7:0] pos,                // Position in sequence
    input wire [7:0] dim,                // Dimension index
    
    output reg signed [`INT32_W-1:0] y0,
    output reg signed [`INT32_W-1:0] y1
);

    // Sin/Cos ROMs for Theta
    // Theta = pos * (10000 ^ (-dim / d_model))
    // We will simulate the ROM with pre-computed (or deterministically generated) values in Q8.8
    reg signed [15:0] sin_rom [0:255][0:31];
    reg signed [15:0] cos_rom [0:255][0:31];

    integer p, d;
    initial begin
        // Simulated initialization. In a real ASIC, these are hardcoded ROMs.
        for (p = 0; p < 256; p = p + 1) begin
            for (d = 0; d < 32; d = d + 1) begin
                // Fake values for synthesis validity
                sin_rom[p][d] = 16'h0080; // 0.5 in Q8.8
                cos_rom[p][d] = 16'h0080; // 0.5 in Q8.8
            end
        end
    end

    wire signed [15:0] sin_theta = sin_rom[pos][dim[4:0]];
    wire signed [15:0] cos_theta = cos_rom[pos][dim[4:0]];

    // Complex multiplication for 2D rotation:
    wire signed [63:0] t0 = (x0 * cos_theta) - (x1 * sin_theta);
    wire signed [63:0] t1 = (x0 * sin_theta) + (x1 * cos_theta);

    always @(posedge clk) begin
        y0 <= t0[39:8];
        y1 <= t1[39:8];
    end

endmodule
