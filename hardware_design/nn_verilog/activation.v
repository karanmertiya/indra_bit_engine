`timescale 1ns / 1ps

// ReLU Activation
module relu_q8_8 (
    input  [15:0] in,
    output [15:0] out
);
    // If sign bit is 1 (negative), output 0, else output input
    assign out = in[15] ? 16'h0000 : in;
endmodule

// ReLU Derivative (for backprop)
module relu_deriv_q8_8 (
    input  [15:0] in,
    output [15:0] out
);
    // Derivative is 1.0 (16'h0100) if >0, else 0.0 (16'h0000)
    assign out = (in[15] || in == 16'h0000) ? 16'h0000 : 16'h0100;
endmodule

// Base-2 Sigmoid Approximation: y = 1 / (1 + 2^-x)
// Implemented via a piecewise linear / LUT approach for Q8.8
module sigmoid_q8_8 (
    input  [15:0] in,
    output reg [15:0] out
);
    wire sign = in[15];
    wire [6:0] int_part = in[14:8];
    wire [7:0] frac_part = in[7:0];
    
    always @(*) begin
        if (sign == 1'b0) begin
            // Positive x
            case (int_part)
                7'd0: out = 16'h0080 + (frac_part >> 1); // 0.5 to ~0.75
                7'd1: out = 16'h00AA + (frac_part >> 2); // 0.66 to ~0.8
                7'd2: out = 16'h00CC + (frac_part >> 3); // 0.8 to ~0.9
                7'd3: out = 16'h00E3 + (frac_part >> 4); // 0.88 to ~0.94
                7'd4: out = 16'h00F0 + (frac_part >> 5); // 0.94 to ~0.97
                default: out = 16'h0100;                 // 1.0 (Saturation)
            endcase
            if (out > 16'h0100) out = 16'h0100; // Cap at 1.0
        end else begin
            // Negative x
            // Magnitude of integer part
            wire [6:0] abs_int = (~int_part) + (frac_part == 0 ? 1'b1 : 1'b0);
            case (abs_int)
                7'd0: out = 16'h0080 - (frac_part >> 1);
                7'd1: out = 16'h0055 - (frac_part >> 2);
                7'd2: out = 16'h0033 - (frac_part >> 3);
                7'd3: out = 16'h001C - (frac_part >> 4);
                7'd4: out = 16'h000F - (frac_part >> 5);
                default: out = 16'h0000;                 // 0.0 (Saturation)
            endcase
        end
    end
endmodule

// Sigmoid Derivative (for backprop): out = sig * (1 - sig)
module sigmoid_deriv_q8_8 (
    input  [15:0] sig_in, // Take the already computed sigmoid output as input
    output [15:0] out
);
    wire [15:0] one_minus_sig = 16'h0100 - sig_in;
    
    // sig * (1 - sig)
    mult_q8_8 deriv_mult (
        .A(sig_in),
        .B(one_minus_sig),
        .P(out)
    );
endmodule
