`timescale 1ns / 1ps

module tb_nn;

    reg clk;
    reg rst;
    reg [47:0] x_in;
    reg [31:0] target;
    reg en_update;
    reg [3:0] lr_shift;
    
    wire [31:0] prediction;

    // Instantiate Top Module
    nn_top uut (
        .clk(clk),
        .rst(rst),
        .x_in(x_in),
        .target(target),
        .en_update(en_update),
        .lr_shift(lr_shift),
        .prediction(prediction)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Test sequence
    initial begin
        // 1. Initialize
        rst = 1;
        en_update = 0;
        x_in = 0;
        target = 0;
        lr_shift = 4'd4; // lr = 1/16
        
        #20 rst = 0;
        
        // 2. Apply input and target
        // Q8.8 format: 1.0 = 16'h0100, -0.5 = 16'hFF80, 0.25 = 16'h0040
        x_in = {16'h0100, 16'hFF80, 16'h0040}; 
        
        // Target: 0.8 = 16'h00CC, 0.2 = 16'h0033
        target = {16'h00CC, 16'h0033};
        
        #20;
        
        // Display Initial Prediction (before training)
        $display("--- INITIAL PREDICTION ---");
        $display("Time: %0t | Pred0: %h | Pred1: %h | Target0: %h | Target1: %h", 
                 $time, prediction[15:0], prediction[31:16], target[15:0], target[31:16]);
                 
        // 3. Enable Training (Backpropagation)
        en_update = 1;
        
        // Let it train for 100 clock cycles
        #1000;
        en_update = 0;
        
        // 4. Display Final Prediction (after training)
        $display("--- FINAL PREDICTION (AFTER 100 EPOCHS) ---");
        $display("Time: %0t | Pred0: %h | Pred1: %h | Target0: %h | Target1: %h", 
                 $time, prediction[15:0], prediction[31:16], target[15:0], target[31:16]);
                 
        #20 $finish;
    end

    // Monitor for changes during training
    initial begin
        $monitor("Time: %0t | Pred0: %h | Pred1: %h", $time, prediction[15:0], prediction[31:16]);
    end

endmodule
