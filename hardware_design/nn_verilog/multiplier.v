`timescale 1ns / 1ps

// Half Adder
module ha (
    input a, b,
    output sum, cout
);
    assign sum = a ^ b;
    assign cout = a & b;
endmodule

// Full Adder
module fa (
    input a, b, cin,
    output sum, cout
);
    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (cin & (a ^ b));
endmodule

// 15x15 Unsigned Array Multiplier
module array_mult_15x15 (
    input  [14:0] A,
    input  [14:0] B,
    output [29:0] P
);
    wire [14:0] pp [14:0];
    wire [14:0] sum [14:0];
    wire [14:0] cout [14:0];

    genvar i, j;
    generate
        // Generate partial products
        for (i = 0; i < 15; i = i + 1) begin : gen_pp_row
            for (j = 0; j < 15; j = j + 1) begin : gen_pp_col
                assign pp[i][j] = A[j] & B[i];
            end
        end

        // First row of addition (just passes the first partial products)
        for (j = 0; j < 15; j = j + 1) begin : gen_row0
            assign sum[0][j] = pp[0][j];
            assign cout[0][j] = 1'b0;
        end

        // Remaining rows of addition
        for (i = 1; i < 15; i = i + 1) begin : gen_add_row
            // First column (LSB of the row)
            ha u_ha (
                .a(pp[i][0]),
                .b(sum[i-1][1]),
                .sum(sum[i][0]),
                .cout(cout[i][0])
            );
            
            // Middle columns
            for (j = 1; j < 14; j = j + 1) begin : gen_add_col
                fa u_fa (
                    .a(pp[i][j]),
                    .b(sum[i-1][j+1]),
                    .cin(cout[i][j-1]),
                    .sum(sum[i][j]),
                    .cout(cout[i][j])
                );
            end

            // Last column (MSB of the row)
            ha u_ha_msb (
                .a(pp[i][14]),
                .b(cout[i][13]),
                .sum(sum[i][14]),
                .cout(cout[i][14])
            );
        end
    endgenerate

    // Assign final product bits
    generate
        for (i = 0; i < 15; i = i + 1) begin : gen_p_out
            assign P[i] = sum[i][0];
        end
        for (j = 1; j < 15; j = j + 1) begin : gen_p_out_msb
            assign P[14+j] = sum[14][j];
        end
        assign P[29] = cout[14][14];
    endgenerate
endmodule

// Signed Q8.8 16x16 Multiplier
// Format: 1 bit sign, 7 bits integer, 8 bits fractional
module mult_q8_8 (
    input  [15:0] A,
    input  [15:0] B,
    output [15:0] P
);
    wire sign_A = A[15];
    wire sign_B = B[15];
    wire sign_P = sign_A ^ sign_B;
    
    // Absolute values
    wire [14:0] abs_A = sign_A ? (~A[14:0] + 1'b1) : A[14:0];
    wire [14:0] abs_B = sign_B ? (~B[14:0] + 1'b1) : B[14:0];
    
    wire [29:0] abs_P;
    
    array_mult_15x15 u_mult (
        .A(abs_A),
        .B(abs_B),
        .P(abs_P)
    );
    
    // Q8.8 output formatting
    // 30-bit result format (Q14.16)
    // We want Q7.8 (15 bits magnitude) which are bits [22:8]
    wire [14:0] mag_P = abs_P[22:8];
    
    // Re-apply sign
    assign P = sign_P ? {1'b1, (~mag_P + 1'b1)} : {1'b0, mag_P};

endmodule
