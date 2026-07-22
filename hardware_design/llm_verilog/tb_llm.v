`timescale 1ns / 1ps
`include "llm_params.vh"

module tb_llm;

    reg clk;
    reg rst;
    reg [`SEQ_LEN*8-1:0] input_tokens;
    wire [`SEQ_LEN*8-1:0] predicted_tokens;

    // Instantiate Top Module
    llm_top uut (
        .clk(clk),
        .rst(rst),
        .input_tokens(input_tokens),
        .predicted_tokens(predicted_tokens)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Test sequence
    integer s;
    initial begin
        // 1. Initialize
        rst = 1;
        input_tokens = 0;
        
        #20 rst = 0;
        
        // 2. Apply a prompt sequence of tokens
        // For example, tokens: [10, 25, 100, 250], rest are 0
        input_tokens[0*8 +: 8] = 8'd10;
        input_tokens[1*8 +: 8] = 8'd25;
        input_tokens[2*8 +: 8] = 8'd100;
        input_tokens[3*8 +: 8] = 8'd250;
        
        #20;
        
        // Wait for combinational logic to settle (in a real pipelined ASIC this would take multiple cycles)
        #100;
        
        // Display Predictions
        $display("--- MICRO-LLM TOY SIMULATION RESULTS ---");
        $display("Time: %0t", $time);
        for (s = 0; s < 4; s = s + 1) begin
            $display("Input Token[%0d]: %3d --> Predicted Next Token: %3d", 
                s, input_tokens[s*8 +: 8], predicted_tokens[s*8 +: 8]);
        end
                 
        #20 $finish;
    end

endmodule
