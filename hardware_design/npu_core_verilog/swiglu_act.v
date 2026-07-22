`timescale 1ns / 1ps
`include "npu_params.vh"

module swiglu_act (
    input wire clk,
    input wire signed [`INT32_W-1:0] x, // Input 1 (Gate)
    input wire signed [`INT32_W-1:0] y, // Input 2 (Data)
    
    output reg signed [`INT32_W-1:0] out
);

    // Step 1: Approximate Sigmoid(x) using piecewise or LUT
    reg signed [15:0] sig_x; // Q8.8
    
    // We assume x is INT32 but we only need the upper significant bits to estimate sigmoid.
    // Let's use an extremely simplified hardware proxy for sigmoid to demonstrate pipeline.
    always @(*) begin
        if (x > 1024) sig_x = 16'h0100; // 1.0 in Q8.8
        else if (x < -1024) sig_x = 16'h0000; // 0.0
        else begin
            // Linear approximation in the active region
            // sig_x = 0.5 + x/2048
            sig_x = 16'h0080 + (x[15:0] >> 3);
            if (sig_x < 0) sig_x = 0;
            if (sig_x > 16'h0100) sig_x = 16'h0100;
        end
    end

    // Step 2 & 3: Swish(x) = x * sig(x), then Out = Swish(x) * y
    wire signed [47:0] swish_x = x * sig_x;
    wire signed [`INT32_W-1:0] swish_x_scaled = swish_x >>> 8; 
    wire signed [63:0] final_mult = swish_x_scaled * y;

    always @(posedge clk) begin
        out <= final_mult[31:0];
    end

endmodule
