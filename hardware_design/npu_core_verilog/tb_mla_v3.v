`timescale 1ns / 1ps

module tb_mla_v3;
    reg clk;
    reg rst_n;
    reg [63:0] hidden_state_in;
    wire [31:0] latent_c_out;
    wire [63:0] key_out;
    wire [63:0] value_out;
    wire out_valid;

    mla_core_v3 uut (
        .clk(clk),
        .rst_n(rst_n),
        .hidden_state_in(hidden_state_in),
        .latent_c_out(latent_c_out),
        .key_out(key_out),
        .value_out(value_out),
        .out_valid(out_valid)
    );

    always #5 clk = ~clk;

    initial begin
        $dumpfile("mla.vcd");
        $dumpvars(0, tb_mla_v3);
        
        clk = 0;
        rst_n = 0;
        hidden_state_in = 0;
        #20;
        rst_n = 1;

        // Test 1: h = [1.0, 1.0, 0.0, 0.0]
        // Since W_down[0] matches the first two indices, latent c[0] should be 1.0
        hidden_state_in[15:0]  = 16'h0100;
        hidden_state_in[31:16] = 16'h0100;
        hidden_state_in[47:32] = 16'h0000;
        hidden_state_in[63:48] = 16'h0000;
        #20;
        
        // Test 2: h = [0.0, 0.0, 1.0, 1.0]
        // Latent c[1] should be 1.0
        hidden_state_in[15:0]  = 16'h0000;
        hidden_state_in[31:16] = 16'h0000;
        hidden_state_in[47:32] = 16'h0100;
        hidden_state_in[63:48] = 16'h0100;
        #50;
        
        $finish;
    end
endmodule
