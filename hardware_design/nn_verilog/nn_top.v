`timescale 1ns / 1ps

module nn_top (
    input wire clk,
    input wire rst,
    input wire [47:0] x_in,       // 3 inputs * 16 bits (Q8.8)
    input wire [31:0] target,     // 2 target outputs * 16 bits (Q8.8)
    input wire en_update,         // Enable learning (backward pass)
    input wire [3:0] lr_shift,    // Learning rate shift
    output wire [31:0] prediction // 2 outputs * 16 bits (Q8.8)
);

    // Wires between layers
    // Layer 1 (Hidden 1): 3 -> 4
    wire [63:0] l1_out;         // 4 * 16
    wire [47:0] l1_err_out;     // 3 * 16 (Not used, since this is the input layer)
    
    // Layer 2 (Hidden 2): 4 -> 4
    wire [63:0] l2_out;         // 4 * 16
    wire [63:0] l2_err_out;     // 4 * 16 (passed to L1 err_in_sum)
    
    // Layer 3 (Output): 4 -> 2
    wire [31:0] l3_out;         // 2 * 16
    wire [63:0] l3_err_out;     // 4 * 16 (passed to L2 err_in_sum)
    
    // Output layer error calculation: (target - prediction)
    wire [15:0] target_arr [0:1];
    wire [15:0] pred_arr [0:1];
    wire [31:0] l3_err_in;      // 2 * 16
    
    assign target_arr[0] = target[15:0];
    assign target_arr[1] = target[31:16];
    assign pred_arr[0] = l3_out[15:0];
    assign pred_arr[1] = l3_out[31:16];
    
    assign l3_err_in[15:0]  = target_arr[0] - pred_arr[0];
    assign l3_err_in[31:16] = target_arr[1] - pred_arr[1];

    assign prediction = l3_out;

    // --- Layer 1 (Hidden 1): 3 inputs, 4 neurons, ReLU ---
    layer #(
        .N_IN(3),
        .N_OUT(4),
        .ACTIVATION_TYPE(0) // 0: ReLU
    ) hidden_layer_1 (
        .clk(clk),
        .rst(rst),
        .x_in(x_in),
        .err_in_sum(l2_err_out), // Error from Layer 2
        .en_update(en_update),
        .lr_shift(lr_shift),
        .y_out(l1_out),
        .err_out_sum(l1_err_out) // Not used
    );

    // --- Layer 2 (Hidden 2): 4 inputs, 4 neurons, ReLU ---
    layer #(
        .N_IN(4),
        .N_OUT(4),
        .ACTIVATION_TYPE(0) // 0: ReLU
    ) hidden_layer_2 (
        .clk(clk),
        .rst(rst),
        .x_in(l1_out),
        .err_in_sum(l3_err_out), // Error from Layer 3
        .en_update(en_update),
        .lr_shift(lr_shift),
        .y_out(l2_out),
        .err_out_sum(l2_err_out)
    );

    // --- Layer 3 (Output): 4 inputs, 2 neurons, Sigmoid ---
    layer #(
        .N_IN(4),
        .N_OUT(2),
        .ACTIVATION_TYPE(1) // 1: Sigmoid
    ) output_layer (
        .clk(clk),
        .rst(rst),
        .x_in(l2_out),
        .err_in_sum(l3_err_in),  // target - prediction
        .en_update(en_update),
        .lr_shift(lr_shift),
        .y_out(l3_out),
        .err_out_sum(l3_err_out)
    );

endmodule
