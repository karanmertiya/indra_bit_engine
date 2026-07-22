`timescale 1ns / 1ps
`include "npu_params.vh"

module mac_cell (
    input wire clk,
    input wire rst,
    input wire drain_en,
    
    input wire signed [`INT8_W-1:0] act_in,     // Activation from left
    input wire signed [`INT8_W-1:0] weight_in,  // Weight from top
    input wire signed [`INT32_W-1:0] psum_in,   // Partial sum from top (used during drain)
    
    output reg signed [`INT8_W-1:0] act_out,    // Activation to right
    output reg signed [`INT8_W-1:0] weight_out, // Weight to bottom
    output wire signed [`INT32_W-1:0] psum_out  // Partial sum to bottom
);

    reg signed [`INT32_W-1:0] psum_reg;
    assign psum_out = psum_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            psum_reg <= 0;
            act_out <= 0;
            weight_out <= 0;
        end else begin
            if (drain_en) begin
                // During drain, the column acts like a vertical shift register
                psum_reg <= psum_in;  // Shift the result from the cell above into our register
            end else begin
                // Normal Computation Phase
                // Multiply and Accumulate into local stationary register
                psum_reg <= psum_reg + (act_in * weight_in);
            end
            
            // Pipeline propagation for wavefront
            act_out <= act_in;
            weight_out <= weight_in;
        end
    end

endmodule
