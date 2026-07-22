`timescale 1ns / 1ps

module rope_vector_unit (
    input wire clk,
    input wire rst_n,
    input wire [511:0] vec_in,
    input wire [7:0] pos,
    input wire in_valid,
    
    output wire [511:0] vec_out,
    output wire out_valid
);

    wire [7:0] valid_out_wires;
    assign out_valid = valid_out_wires[0];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : gen_rope
            rope_act_v3 u_rope (
                .clk(clk),
                .rst_n(rst_n),
                .x0(vec_in[(i*2)*32 +: 32]),
                .x1(vec_in[(i*2+1)*32 +: 32]),
                .pos(pos),
                .valid_in(in_valid),
                .dim(6'(i * 2)), // Dimension index
                .y0(vec_out[(i*2)*32 +: 32]),
                .y1(vec_out[(i*2+1)*32 +: 32]),
                .out_valid(valid_out_wires[i])
            );
        end
    endgenerate

endmodule
