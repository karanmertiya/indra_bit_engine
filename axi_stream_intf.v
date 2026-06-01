`timescale 1ns / 1ps
`include "npu_params.vh"

module axi_stream_intf (
    input wire aclk,
    input wire aresetn,

    // AXI-Stream Slave Interface (Simultaneous Weights & Activations)
    // [127:0] Acts, [255:128] Weights
    input wire [255:0] s_axis_tdata, 
    input wire s_axis_tvalid,
    input wire s_axis_tlast,
    output wire s_axis_tready,

    // AXI-Stream Master Interface (Outbound Results)
    output reg [511:0] m_axis_tdata,
    output reg m_axis_tvalid,
    output reg m_axis_tlast,
    input wire m_axis_tready,
    
    // Interface to Systolic Array
    output reg array_drain_en,
    output reg [`ARRAY_SIZE*`INT8_W-1:0] array_weight_in,
    output reg [`ARRAY_SIZE*`INT8_W-1:0] array_act_in,
    input wire [`ARRAY_SIZE*`INT32_W-1:0] array_psum_out
);

    reg [5:0] compute_cycles;
    reg [4:0] drain_cycles;
    
    // States
    localparam IDLE = 0, COMPUTE = 1, DRAIN = 2;
    reg [1:0] state;
    
    assign s_axis_tready = (state == IDLE || state == COMPUTE) && m_axis_tready;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            array_drain_en <= 0;
            array_weight_in <= 0;
            array_act_in <= 0;
            m_axis_tvalid <= 0;
            m_axis_tdata <= 0;
            m_axis_tlast <= 0;
            compute_cycles <= 0;
            drain_cycles <= 0;
            state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    array_drain_en <= 0;
                    m_axis_tvalid <= 0;
                    m_axis_tlast <= 0;
                    if (s_axis_tvalid && s_axis_tready) begin
                        array_act_in <= s_axis_tdata[127:0];
                        array_weight_in <= s_axis_tdata[255:128];
                        compute_cycles <= 1;
                        state <= COMPUTE;
                    end else begin
                        array_act_in <= 0;
                        array_weight_in <= 0;
                    end
                end
                
                COMPUTE: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        array_act_in <= s_axis_tdata[127:0];
                        array_weight_in <= s_axis_tdata[255:128];
                        compute_cycles <= compute_cycles + 1;
                        
                        // Takes 32 cycles to fully flush a 16x16 matrix wave
                        if (compute_cycles == 31) begin
                            state <= DRAIN;
                            drain_cycles <= 0;
                            array_drain_en <= 1;
                        end
                    end else begin
                        array_act_in <= 0;
                        array_weight_in <= 0;
                    end
                end
                
                DRAIN: begin
                    array_act_in <= 0;
                    array_weight_in <= 0;
                    
                    if (m_axis_tready) begin
                        m_axis_tvalid <= 1;
                        m_axis_tdata <= array_psum_out;
                        
                        drain_cycles <= drain_cycles + 1;
                        if (drain_cycles == 15) begin
                            m_axis_tlast <= 1;
                            state <= IDLE; // Finished draining 16 rows
                            array_drain_en <= 0;
                        end else begin
                            m_axis_tlast <= 0;
                        end
                    end
                end
            endcase
        end
    end
endmodule
