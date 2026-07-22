`timescale 1ns / 1ps

module tb_rope_v3;
    reg clk;
    reg rst_n;
    reg signed [31:0] x0;
    reg signed [31:0] x1;
    reg [7:0] pos;
    reg [5:0] dim;
    wire signed [31:0] y0;
    wire signed [31:0] y1;
    wire out_valid;

    rope_act_v3 uut (
        .clk(clk),
        .rst_n(rst_n),
        .x0(x0),
        .x1(x1),
        .pos(pos),
        .dim(dim),
        .y0(y0),
        .y1(y1),
        .out_valid(out_valid)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("rope.vcd");
        $dumpvars(0, tb_rope_v3);
        
        clk = 0;
        rst_n = 0;
        x0 = 0;
        x1 = 0;
        pos = 0;
        dim = 0;
        #20;
        rst_n = 1;

        // Test 1: pos=0, dim=0 (Theta = 0)
        // Cos=1.0, Sin=0.0 => y0=x0, y1=x1
        x0 = 32'h0001_0000; // 1.0
        x1 = 32'h0000_0000; // 0.0
        pos = 0;
        dim = 0;
        #10;
        
        // Test 2: pos=4, dim=0 (Theta = 90 deg)
        // Cos=0.0, Sin=1.0 => y0=-x1, y1=x0
        x0 = 32'h0001_0000;
        x1 = 32'h0000_0000;
        pos = 4;
        dim = 0;
        #10;
        
        // Test 3: Mixed positions
        x0 = 32'd100000;
        x1 = 32'd50000;
        pos = 2;
        dim = 1;
        #50;
        
        $finish;
    end
endmodule
