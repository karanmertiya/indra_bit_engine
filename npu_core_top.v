`timescale 1ns / 1ps
`include "npu_params.vh"

module npu_core_top (
    input wire aclk,
    input wire aresetn,

    // AXI-Stream Inbound (256-bit: [127:0] Acts, [255:128] Weights)
    input wire [255:0] s_axis_tdata,
    input wire s_axis_tvalid,
    input wire s_axis_tlast,
    output wire s_axis_tready,

    // AXI-Stream Outbound (Results)
    output wire [511:0] m_axis_tdata,
    output wire m_axis_tvalid,
    output wire m_axis_tlast,
    input wire m_axis_tready
);

    // Wires for Systolic Array Connection
    wire array_drain_en;
    wire [`ARRAY_SIZE*`INT8_W-1:0] array_weight_in;
    wire [`ARRAY_SIZE*`INT8_W-1:0] array_act_in;
    wire [`ARRAY_SIZE*`INT32_W-1:0] array_psum_out;

    // 1. AXI-Stream Interface
    axi_stream_intf u_axis_intf (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tready(m_axis_tready),
        
        .array_drain_en(array_drain_en),
        .array_weight_in(array_weight_in),
        .array_act_in(array_act_in),
        .array_psum_out(array_psum_out)
    );

    // 2. 16x16 Output-Stationary Systolic Array
    systolic_array_16x16 u_systolic_array (
        .clk(aclk),
        .rst(~aresetn),
        .drain_en(array_drain_en),
        .act_in_vec(array_act_in),
        .weight_in_vec(array_weight_in),
        .psum_out_vec(array_psum_out)
    );
    
    // 3. RoPE Module Example (Unchanged but using new datapath)
    wire signed [`INT32_W-1:0] rope_y0, rope_y1;
    rope_rotate u_rope (
        .clk(aclk),
        .x0(array_psum_out[31:0]),
        .x1(array_psum_out[63:32]),
        .pos(8'd0), 
        .dim(8'd0), 
        .y0(rope_y0),
        .y1(rope_y1)
    );
    
    // 4. SwiGLU Module Example (Unchanged)
    wire signed [`INT32_W-1:0] swi_out;
    swiglu_act u_swiglu (
        .clk(aclk),
        .x(array_psum_out[95:64]),
        .y(array_psum_out[127:96]),
        .out(swi_out)
    );

endmodule
