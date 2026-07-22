`timescale 1ns / 1ps
`include "npu_params.vh"

module systolic_array_16x16 (
    input wire clk,
    input wire rst,
    input wire drain_en,
    
    // Activation Stream Interface (Left)
    input wire [`ARRAY_SIZE*`INT8_W-1:0] act_in_vec, // 16 activations
    
    // Weight Stream Interface (Top)
    input wire [`ARRAY_SIZE*`INT8_W-1:0] weight_in_vec, // 16 weights
    
    // Output Partial Sums (Bottom)
    output wire [`ARRAY_SIZE*`INT32_W-1:0] psum_out_vec // 16 resulting partial sums
);

    // Internal wires interconnecting the 16x16 grid
    wire [`INT8_W-1:0] act_wire [0:`ARRAY_SIZE-1][0:`ARRAY_SIZE];     // [row][col]
    wire [`INT8_W-1:0] weight_wire [0:`ARRAY_SIZE][0:`ARRAY_SIZE-1];  // [row][col]
    wire [`INT32_W-1:0] psum_wire [0:`ARRAY_SIZE][0:`ARRAY_SIZE-1];   // [row][col]
    
    // Unpack inputs
    genvar i, j;
    generate
        // Connect acts to left edge
        for (i = 0; i < `ARRAY_SIZE; i = i + 1) begin : gen_act_in
            assign act_wire[i][0] = act_in_vec[i*`INT8_W +: `INT8_W];
        end
        // Connect weights to top edge
        for (j = 0; j < `ARRAY_SIZE; j = j + 1) begin : gen_weight_in
            assign weight_wire[0][j] = weight_in_vec[j*`INT8_W +: `INT8_W];
        end
        // Initialize top edge psums to 0 (used during draining)
        for (j = 0; j < `ARRAY_SIZE; j = j + 1) begin : gen_psum_in
            assign psum_wire[0][j] = 32'd0;
        end
        // Pack bottom edge psums to output
        for (j = 0; j < `ARRAY_SIZE; j = j + 1) begin : gen_psum_out
            assign psum_out_vec[j*`INT32_W +: `INT32_W] = psum_wire[`ARRAY_SIZE][j];
        end
    endgenerate

    // Instantiate 16x16 MAC Cells
    generate
        for (i = 0; i < `ARRAY_SIZE; i = i + 1) begin : row
            for (j = 0; j < `ARRAY_SIZE; j = j + 1) begin : col
                mac_cell u_mac (
                    .clk(clk),
                    .rst(rst),
                    .drain_en(drain_en),
                    .act_in(act_wire[i][j]),
                    .weight_in(weight_wire[i][j]),
                    .psum_in(psum_wire[i][j]),
                    .act_out(act_wire[i][j+1]),
                    .weight_out(weight_wire[i+1][j]),
                    .psum_out(psum_wire[i+1][j])
                );
            end
        end
    endgenerate

endmodule
