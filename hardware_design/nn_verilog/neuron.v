`timescale 1ns / 1ps

module neuron #(
    parameter N_IN = 3
)(
    input wire clk,
    input wire rst,
    input wire [16*N_IN-1:0] x_in,   // Flattened inputs
    input wire [15:0] error_in,      // Backpropagated error for this neuron
    input wire en_update,            // Enable weight update
    input wire [3:0] lr_shift,       // Learning rate shift (e.g., 2 for >>2)
    output reg [15:0] pre_act,       // Pre-activation output (MAC sum)
    output wire [16*N_IN-1:0] w_out  // Expose weights for backprop to previous layer
);

    reg [15:0] weights [0:N_IN-1];
    wire [15:0] x_arr [0:N_IN-1];
    wire [15:0] mult_out [0:N_IN-1];
    wire [15:0] update_mult [0:N_IN-1];

    genvar i;
    generate
        for (i = 0; i < N_IN; i = i + 1) begin : gen_inputs
            assign x_arr[i] = x_in[16*i +: 16];
            assign w_out[16*i +: 16] = weights[i];

            // Forward pass multipliers
            mult_q8_8 fwd_mult (
                .A(x_arr[i]),
                .B(weights[i]),
                .P(mult_out[i])
            );

            // Backward pass multiplier: error * input
            mult_q8_8 bwd_mult (
                .A(error_in),
                .B(x_arr[i]),
                .P(update_mult[i])
            );
        end
    endgenerate

    // Adder tree for forward pass
    // Simple behavioral sum for MAC (synthesizes to adder tree)
    integer j;
    always @(*) begin
        pre_act = 16'h0000;
        for (j = 0; j < N_IN; j = j + 1) begin
            pre_act = pre_act + mult_out[j];
        end
    end

    // Synchronous Weight Update using D-Flip-Flops
    integer k;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (k = 0; k < N_IN; k = k + 1) begin
                // Initialize with some small random value or 0
                // Using a generic 0.1 (0x0019) for initialization
                weights[k] <= 16'h0019; 
            end
        end else if (en_update) begin
            for (k = 0; k < N_IN; k = k + 1) begin
                // W_new = W_old + (error * input >> lr_shift)
                // Assuming error_in is correctly signed
                weights[k] <= weights[k] + (update_mult[k] >>> lr_shift);
            end
        end
    end

endmodule
