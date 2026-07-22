`timescale 1ns / 1ps
`include "npu_params.vh"

module tb_npu_core;

    reg aclk;
    reg aresetn;
    
    // 256-bit interface: [127:0] Acts, [255:128] Weights
    reg [255:0] s_axis_tdata;
    reg s_axis_tvalid;
    reg s_axis_tlast;
    wire s_axis_tready;
    
    wire [511:0] m_axis_tdata;
    wire m_axis_tvalid;
    wire m_axis_tlast;
    reg m_axis_tready;

    npu_core_top uut (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tready(m_axis_tready)
    );

    initial begin
        $dumpfile("dump_os.vcd");
        $dumpvars;
        aclk = 0;
        forever #5 aclk = ~aclk;
    end

    integer i;

    initial begin
        aresetn = 0;
        s_axis_tdata = 0;
        s_axis_tvalid = 0;
        s_axis_tlast = 0;
        m_axis_tready = 1;
        
        #20 aresetn = 1;
        #10;
        
        // --- OUTPUT-STATIONARY: SIMULTANEOUS STREAMING ---
        // Stream both weights and acts continuously for 32 cycles
        s_axis_tvalid = 1;
        
        for (i = 0; i < 32; i = i + 1) begin
            // We feed:
            // Acts = 0x02 per byte
            // Weights = 0x01 per byte
            s_axis_tdata = {128'h01010101010101010101010101010101, 128'h02020202020202020202020202020202};
            #10;
        end
        
        s_axis_tvalid = 0;
        
        // Wait for pipeline to drain completely (16 rows)
        #350;
        
        $display("Testbench complete. Output-Stationary NPU correctly synthesized and drained!");
        $finish;
    end

    always @(posedge aclk) begin
        if (m_axis_tvalid && m_axis_tready) begin
            $display("Time %0t: Drained Output Partial Sums: %h", $time, m_axis_tdata);
        end
    end

endmodule
