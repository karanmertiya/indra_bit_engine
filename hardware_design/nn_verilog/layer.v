`timescale 1ns / 1ps

module layer #(
    parameter N_IN = 3,
    parameter N_OUT = 4,
    parameter ACTIVATION_TYPE = 0 // 0 for ReLU, 1 for Sigmoid
)(
    input wire clk,
    input wire rst,
    input wire [16*N_IN-1:0] x_in,         // Activations from prev layer
    input wire [16*N_OUT-1:0] err_in_sum,  // Sum of err * W from next layer
    input wire en_update,
    input wire [3:0] lr_shift,
    output wire [16*N_OUT-1:0] y_out,      // Activations to next layer
    output reg [16*N_IN-1:0] err_out_sum   // Sum of err * W to prev layer
);

    wire [15:0] pre_act [0:N_OUT-1];
    wire [15:0] act_out [0:N_OUT-1];
    wire [15:0] deriv_out [0:N_OUT-1];
    wire [15:0] delta [0:N_OUT-1];
    wire [16*N_IN-1:0] w_out [0:N_OUT-1];
    
    wire [15:0] err_in_arr [0:N_OUT-1];

    genvar i, j;
    generate
        for (i = 0; i < N_OUT; i = i + 1) begin : gen_neurons
            assign err_in_arr[i] = err_in_sum[16*i +: 16];

            // 1. Activation function
            if (ACTIVATION_TYPE == 0) begin : gen_relu
                relu_q8_8 u_relu (
                    .in(pre_act[i]),
                    .out(act_out[i])
                );
                relu_deriv_q8_8 u_relu_deriv (
                    .in(pre_act[i]),
                    .out(deriv_out[i])
                );
            end else begin : gen_sigmoid
                sigmoid_q8_8 u_sigmoid (
                    .in(pre_act[i]),
                    .out(act_out[i])
                );
                sigmoid_deriv_q8_8 u_sigmoid_deriv (
                    .sig_in(act_out[i]), // Sigmoid deriv uses activation output
                    .out(deriv_out[i])
                );
            end

            assign y_out[16*i +: 16] = act_out[i];

            // 2. Delta calculation: delta = err_in_sum * deriv
            mult_q8_8 delta_mult (
                .A(err_in_arr[i]),
                .B(deriv_out[i]),
                .P(delta[i])
            );

            // 3. Neuron instance
            neuron #(
                .N_IN(N_IN)
            ) u_neuron (
                .clk(clk),
                .rst(rst),
                .x_in(x_in),
                .error_in(delta[i]),
                .en_update(en_update),
                .lr_shift(lr_shift),
                .pre_act(pre_act[i]),
                .w_out(w_out[i])
            );
        end
    endgenerate

    // 4. Calculate error to pass to previous layer
    // err_out_sum[j] = sum(delta[i] * w_out[i][j])
    integer r, c;
    reg [15:0] w_tmp;
    wire [15:0] backprop_mult [0:N_IN-1][0:N_OUT-1];

    generate
        for (j = 0; j < N_IN; j = j + 1) begin : gen_err_out
            for (i = 0; i < N_OUT; i = i + 1) begin : gen_bp_mult
                wire [15:0] w_ji = w_out[i][16*j +: 16];
                mult_q8_8 bp_mult (
                    .A(delta[i]),
                    .B(w_ji),
                    .P(backprop_mult[j][i])
                );
            end
        end
    endgenerate

    always @(*) begin
        for (r = 0; r < N_IN; r = r + 1) begin
            err_out_sum[16*r +: 16] = 16'h0000;
            for (c = 0; c < N_OUT; c = c + 1) begin
                err_out_sum[16*r +: 16] = err_out_sum[16*r +: 16] + backprop_mult[r][c];
            end
        end
    end

endmodule
