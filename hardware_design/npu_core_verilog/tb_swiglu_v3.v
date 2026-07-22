`timescale 1ns / 1ps

module tb_swiglu_v3;
    reg clk;
    reg rst_n;
    reg signed [31:0] x;
    reg signed [31:0] y;
    wire signed [31:0] out;
    wire out_valid;

    swiglu_act_v3 uut (
        .clk(clk),
        .rst_n(rst_n),
        .x(x),
        .y(y),
        .out(out),
        .out_valid(out_valid)
    );

    // Clock gen
    always #5 clk = ~clk;

    initial begin
        $dumpfile("swiglu.vcd");
        $dumpvars(0, tb_swiglu_v3);
        
        clk = 0;
        rst_n = 0;
        x = 0;
        y = 0;
        #20;
        rst_n = 1;

        // Test vectors (Q16.16)
        // Let x = 1.0 (65536), y = 2.0 (131072)
        x = 32'h0001_0000;
        y = 32'h0002_0000;
        #10;
        
        // Let x = -1.0 (-65536), y = 1.5 (98304)
        x = -32'sd65536;
        y = 32'd98304;
        #10;
        
        // Let x = 4.0 (262144), y = 0.5 (32768)
        x = 32'd262144;
        y = 32'd32768;
        #10;
        
        // Let x = -4.0 (-262144), y = 1.0 (65536)
        x = -32'sd262144;
        y = 32'd65536;
        #50;
        
        $finish;
    end
endmodule
