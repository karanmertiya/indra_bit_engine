`timescale 1ns / 1ps

module tb_moe_v3;
    reg clk;
    reg rst_n;
    reg [63:0] token_vec_in;
    wire [1:0] selected_expert;
    wire out_valid;

    moe_router_v3 uut (
        .clk(clk),
        .rst_n(rst_n),
        .token_vec_in(token_vec_in),
        .selected_expert(selected_expert),
        .out_valid(out_valid)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("moe.vcd");
        $dumpvars(0, tb_moe_v3);
        
        clk = 0;
        rst_n = 0;
        token_vec_in = 0;
        #20;
        rst_n = 1;

        // Test 1: Route to Expert 0 
        // Expert 0 is [1.0, 0.0, 0.5, -0.5]
        // Send token exactly matching Expert 0: [1.0, 0.0, 0.5, -0.5]
        token_vec_in[15:0]  = 16'h0100; // 1.0
        token_vec_in[31:16] = 16'h0000; // 0.0
        token_vec_in[47:32] = 16'h0080; // 0.5
        token_vec_in[63:48] = -16'h0080; // -0.5
        #10;
        
        // Test 2: Route to Expert 1
        // Expert 1 is [0.0, 1.0, -0.5, 0.5]
        token_vec_in[15:0]  = 16'h0000;
        token_vec_in[31:16] = 16'h0100;
        token_vec_in[47:32] = -16'h0080;
        token_vec_in[63:48] = 16'h0080;
        #10;
        
        // Test 3: Route to Expert 3
        token_vec_in[15:0]  = 16'h0000;
        token_vec_in[31:16] = -16'h0100;
        token_vec_in[47:32] = 16'h0000;
        token_vec_in[63:48] = 16'h0100;
        #40;
        
        $finish;
    end
endmodule
