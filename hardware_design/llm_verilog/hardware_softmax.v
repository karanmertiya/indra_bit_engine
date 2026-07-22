`timescale 1ns / 1ps
`include "llm_params.vh"

// Hardware Softmax over a vector of N elements
// Uses base-2 approximation: 2^(x - max(x)) / sum(2^(x - max(x)))
module hardware_softmax #(
    parameter N = `SEQ_LEN
)(
    input wire clk,
    input wire [N*`DATA_W-1:0] in_vec,
    output reg [N*`DATA_W-1:0] out_vec
);

    wire signed [`DATA_W-1:0] in_arr [0:N-1];
    reg signed [`DATA_W-1:0] max_val;
    reg [`DATA_W-1:0] exp_arr [0:N-1];
    reg [`DATA_W+4:0] sum_exp; // Extra bits for accumulation to prevent overflow
    
    integer i;
    
    // 1. Unpack input and find MAX
    always @(*) begin
        max_val = -32768; // Min 16-bit signed
        for (i = 0; i < N; i = i + 1) begin
            in_arr[i] = in_vec[i*`DATA_W +: `DATA_W];
            if (in_arr[i] > max_val) begin
                max_val = in_arr[i];
            end
        end
    end

    // 2. Subtract MAX and compute Base-2 Exponent (2^x approx)
    // For x <= 0. If x = -1 (in Q8.8, this is -256), 2^-1 = 0.5 (128).
    // We can approximate 2^x by shifting 1.0 (256 in Q8.8) to the right by the integer magnitude of x.
    always @(*) begin
        sum_exp = 0;
        for (i = 0; i < N; i = i + 1) begin
            // diff is always <= 0
            automatic reg signed [`DATA_W-1:0] diff = in_arr[i] - max_val;
            // Get integer part of the negative difference (absolute value)
            // e.g. diff = -2.5 -> int part is 2 or 3.
            automatic reg [7:0] abs_int_diff = (-diff) >> `FRAC_W; 
            
            // 2^diff = 1.0 >> abs_int_diff
            if (abs_int_diff > 8) begin
                exp_arr[i] = 16'h0000; // Underflow to 0
            end else begin
                // 16'h0100 is 1.0 in Q8.8
                exp_arr[i] = 16'h0100 >> abs_int_diff;
            end
            
            sum_exp = sum_exp + exp_arr[i];
        end
    end

    // 3. Division (y_i / sum)
    // Hardware division is slow. A real ASIC uses a reciprocal LUT: 1/sum_exp
    // For this toy model, we will use the division operator which synthesizes to a standard divider.
    always @(*) begin
        for (i = 0; i < N; i = i + 1) begin
            // To maintain Q8.8 precision during division: (y_i << 8) / sum
            if (sum_exp == 0) begin
                out_vec[i*`DATA_W +: `DATA_W] = 16'h0000;
            end else begin
                out_vec[i*`DATA_W +: `DATA_W] = (exp_arr[i] << `FRAC_W) / sum_exp;
            end
        end
    end

endmodule
