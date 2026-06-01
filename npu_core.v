`timescale 1ns / 1ps

`define ARRAY_SIZE 16
`define INT8_W 8
`define INT32_W 32

// --- MAC CELL (OUTPUT-STATIONARY) ---
module mac_cell (
    input wire clk, rst, drain_en,
    input wire signed [`INT8_W-1:0] act_in, weight_in,  
    input wire signed [`INT32_W-1:0] psum_in,   
    output reg signed [`INT8_W-1:0] act_out, weight_out, 
    output reg signed [`INT32_W-1:0] psum_out   
);
    reg signed [`INT32_W-1:0] psum_reg;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            psum_reg <= 0; act_out <= 0; weight_out <= 0; psum_out <= 0;
        end else begin
            if (drain_en) begin
                psum_out <= psum_reg; psum_reg <= psum_in;  
            end else begin
                psum_reg <= psum_reg + (act_in * weight_in); psum_out <= 0; 
            end
            act_out <= act_in; weight_out <= weight_in;
        end
    end
endmodule

// --- SYSTOLIC ARRAY ---
module systolic_array_16x16 (
    input wire clk, rst, drain_en,
    input wire [`ARRAY_SIZE*`INT8_W-1:0] act_in_vec, weight_in_vec, 
    output wire [`ARRAY_SIZE*`INT32_W-1:0] psum_out_vec 
);
    wire [`INT8_W-1:0] act_wire [0:`ARRAY_SIZE-1][0:`ARRAY_SIZE];     
    wire [`INT8_W-1:0] weight_wire [0:`ARRAY_SIZE][0:`ARRAY_SIZE-1];  
    wire [`INT32_W-1:0] psum_wire [0:`ARRAY_SIZE][0:`ARRAY_SIZE-1];   
    
    genvar i, j;
    generate
        for (i = 0; i < `ARRAY_SIZE; i = i + 1) begin : gen_act_in
            assign act_wire[i][0] = act_in_vec[i*`INT8_W +: `INT8_W];
        end
        for (j = 0; j < `ARRAY_SIZE; j = j + 1) begin : gen_weight_in
            assign weight_wire[0][j] = weight_in_vec[j*`INT8_W +: `INT8_W];
            assign psum_wire[0][j] = 32'd0;
            assign psum_out_vec[j*`INT32_W +: `INT32_W] = psum_wire[`ARRAY_SIZE][j];
        end
        for (i = 0; i < `ARRAY_SIZE; i = i + 1) begin : row
            for (j = 0; j < `ARRAY_SIZE; j = j + 1) begin : col
                mac_cell u_mac (
                    .clk(clk), .rst(rst), .drain_en(drain_en),
                    .act_in(act_wire[i][j]), .weight_in(weight_wire[i][j]), .psum_in(psum_wire[i][j]),
                    .act_out(act_wire[i][j+1]), .weight_out(weight_wire[i+1][j]), .psum_out(psum_wire[i+1][j])
                );
            end
        end
    endgenerate
endmodule

// --- AXI STREAM INTERFACE ---
module axi_stream_intf (
    input wire aclk, aresetn,
    input wire [255:0] s_axis_tdata, 
    input wire s_axis_tvalid, s_axis_tlast,
    output wire s_axis_tready,
    output reg [511:0] m_axis_tdata,
    output reg m_axis_tvalid, m_axis_tlast,
    input wire m_axis_tready,
    output reg array_drain_en,
    output reg [`ARRAY_SIZE*`INT8_W-1:0] array_weight_in, array_act_in,
    input wire [`ARRAY_SIZE*`INT32_W-1:0] array_psum_out
);
    reg [5:0] compute_cycles;
    reg [4:0] drain_cycles;
    localparam IDLE = 0, COMPUTE = 1, DRAIN = 2;
    reg [1:0] state;
    
    assign s_axis_tready = (state == IDLE || state == COMPUTE) && m_axis_tready;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            array_drain_en <= 0; array_weight_in <= 0; array_act_in <= 0;
            m_axis_tvalid <= 0; m_axis_tdata <= 0; m_axis_tlast <= 0;
            compute_cycles <= 0; drain_cycles <= 0; state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    array_drain_en <= 0; m_axis_tvalid <= 0; m_axis_tlast <= 0;
                    if (s_axis_tvalid && s_axis_tready) begin
                        array_act_in <= s_axis_tdata[127:0]; array_weight_in <= s_axis_tdata[255:128];
                        compute_cycles <= 1; state <= COMPUTE;
                    end else begin
                        array_act_in <= 0; array_weight_in <= 0;
                    end
                end
                COMPUTE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        array_act_in <= s_axis_tdata[127:0]; array_weight_in <= s_axis_tdata[255:128];
                        compute_cycles <= compute_cycles + 1;
                        if (compute_cycles == 31) begin
                            state <= DRAIN; drain_cycles <= 0; array_drain_en <= 1;
                        end
                    end else begin
                        array_act_in <= 0; array_weight_in <= 0;
                    end
                end
                DRAIN: begin
                    array_act_in <= 0; array_weight_in <= 0;
                    if (m_axis_tready) begin
                        m_axis_tvalid <= 1; m_axis_tdata <= array_psum_out;
                        drain_cycles <= drain_cycles + 1;
                        if (drain_cycles == 15) begin
                            m_axis_tlast <= 1; state <= IDLE; array_drain_en <= 0;
                        end else begin
                            m_axis_tlast <= 0;
                        end
                    end
                end
            endcase
        end
    end
endmodule

// --- NPU CORE TOP MODULE ---
module npu_core_top (
    input wire aclk, aresetn,
    input wire [255:0] s_axis_tdata,
    input wire s_axis_tvalid, s_axis_tlast,
    output wire s_axis_tready,
    output wire [511:0] m_axis_tdata,
    output wire m_axis_tvalid, m_axis_tlast,
    input wire m_axis_tready
);
    wire array_drain_en;
    wire [`ARRAY_SIZE*`INT8_W-1:0] array_weight_in, array_act_in;
    wire [`ARRAY_SIZE*`INT32_W-1:0] array_psum_out;

    axi_stream_intf u_axis_intf (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast), .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast), .m_axis_tready(m_axis_tready),
        .array_drain_en(array_drain_en), .array_weight_in(array_weight_in),
        .array_act_in(array_act_in), .array_psum_out(array_psum_out)
    );

    systolic_array_16x16 u_systolic_array (
        .clk(aclk), .rst(~aresetn), .drain_en(array_drain_en),
        .act_in_vec(array_act_in), .weight_in_vec(array_weight_in),
        .psum_out_vec(array_psum_out)
    );
endmodule
