`timescale 1ns / 1ps
`include "llm_params.vh"

module attention (
    input wire clk,
    input wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] x_in, // Input sequence
    output reg [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] out  // Attended output
);

    // Internal arrays for easier indexing
    reg signed [`DATA_W-1:0] X [0:`SEQ_LEN-1][0:`EMBED_DIM-1];
    reg signed [`DATA_W-1:0] Q [0:`NUM_HEADS-1][0:`SEQ_LEN-1][0:`HEAD_DIM-1];
    reg signed [`DATA_W-1:0] K [0:`NUM_HEADS-1][0:`SEQ_LEN-1][0:`HEAD_DIM-1];
    reg signed [`DATA_W-1:0] V [0:`NUM_HEADS-1][0:`SEQ_LEN-1][0:`HEAD_DIM-1];
    
    // Weights (Simulated ROM for inference)
    reg signed [`DATA_W-1:0] Wq [0:`EMBED_DIM-1][0:`EMBED_DIM-1];
    reg signed [`DATA_W-1:0] Wk [0:`EMBED_DIM-1][0:`EMBED_DIM-1];
    reg signed [`DATA_W-1:0] Wv [0:`EMBED_DIM-1][0:`EMBED_DIM-1];
    reg signed [`DATA_W-1:0] Wo [0:`EMBED_DIM-1][0:`EMBED_DIM-1];

    // Initialization of weights
    integer i, j, s, h, d, s2;
    initial begin
        for (i = 0; i < `EMBED_DIM; i = i + 1) begin
            for (j = 0; j < `EMBED_DIM; j = j + 1) begin
                // Identity-like or small random weights (shifted to Q8.8)
                Wq[i][j] = (i == j) ? 16'h0100 : 16'h0000;
                Wk[i][j] = (i == j) ? 16'h0100 : 16'h0000;
                Wv[i][j] = (i == j) ? 16'h0100 : 16'h0000;
                Wo[i][j] = (i == j) ? 16'h0100 : 16'h0000;
            end
        end
    end

    // Temporary variables for Matrix Multiplications
    reg signed [31:0] acc;
    reg signed [`DATA_W-1:0] scores [0:`NUM_HEADS-1][0:`SEQ_LEN-1][0:`SEQ_LEN-1];
    
    // Flattened arrays for Softmax module instantiations
    wire [`SEQ_LEN*`DATA_W-1:0] sm_in [0:`NUM_HEADS-1][0:`SEQ_LEN-1];
    wire [`SEQ_LEN*`DATA_W-1:0] sm_out [0:`NUM_HEADS-1][0:`SEQ_LEN-1];
    
    // Instantiate Softmax modules (One per row per head)
    genvar gh, gs;
    generate
        for (gh = 0; gh < `NUM_HEADS; gh = gh + 1) begin : gen_head
            for (gs = 0; gs < `SEQ_LEN; gs = gs + 1) begin : gen_seq
                hardware_softmax u_sm (
                    .clk(clk),
                    .in_vec(sm_in[gh][gs]),
                    .out_vec(sm_out[gh][gs])
                );
            end
        end
    endgenerate

    reg signed [`DATA_W-1:0] probs [0:`NUM_HEADS-1][0:`SEQ_LEN-1][0:`SEQ_LEN-1];
    reg signed [`DATA_W-1:0] head_out [0:`NUM_HEADS-1][0:`SEQ_LEN-1][0:`HEAD_DIM-1];
    reg signed [`DATA_W-1:0] concat_out [0:`SEQ_LEN-1][0:`EMBED_DIM-1];

    always @(*) begin
        // 1. Unpack Input
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                X[s][i] = x_in[(s*`EMBED_DIM + i)*`DATA_W +: `DATA_W];
            end
        end

        // 2. Compute Q, K, V (X * W)
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (h = 0; h < `NUM_HEADS; h = h + 1) begin
                for (d = 0; d < `HEAD_DIM; d = d + 1) begin
                    acc = 0;
                    for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                        acc = acc + (X[s][i] * Wq[i][h*`HEAD_DIM + d]);
                    end
                    Q[h][s][d] = acc >>> `FRAC_W;
                    
                    acc = 0;
                    for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                        acc = acc + (X[s][i] * Wk[i][h*`HEAD_DIM + d]);
                    end
                    K[h][s][d] = acc >>> `FRAC_W;
                    
                    acc = 0;
                    for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                        acc = acc + (X[s][i] * Wv[i][h*`HEAD_DIM + d]);
                    end
                    V[h][s][d] = acc >>> `FRAC_W;
                end
            end
        end

        // 3. Compute Attention Scores (Q * K^T) and Masking
        for (h = 0; h < `NUM_HEADS; h = h + 1) begin
            for (s = 0; s < `SEQ_LEN; s = s + 1) begin
                for (s2 = 0; s2 < `SEQ_LEN; s2 = s2 + 1) begin
                    if (s2 > s) begin
                        // Causal Masking
                        scores[h][s][s2] = -32768; // -Infinity
                    end else begin
                        acc = 0;
                        for (d = 0; d < `HEAD_DIM; d = d + 1) begin
                            acc = acc + (Q[h][s][d] * K[h][s2][d]);
                        end
                        // Scaling by sqrt(HEAD_DIM). If HEAD_DIM=8, sqrt=2.8.
                        // We divide by 2 for simplicity (shift right by 1)
                        scores[h][s][s2] = (acc >>> `FRAC_W) >>> 1;
                    end
                end
            end
        end
    end

    // 4. Pack scores to Softmax module inputs
    generate
        for (gh = 0; gh < `NUM_HEADS; gh = gh + 1) begin : pack_head
            for (gs = 0; gs < `SEQ_LEN; gs = gs + 1) begin : pack_seq
                wire [`SEQ_LEN*`DATA_W-1:0] temp_sm_in;
                genvar gs2;
                for (gs2 = 0; gs2 < `SEQ_LEN; gs2 = gs2 + 1) begin : pack_s2
                    assign temp_sm_in[gs2*`DATA_W +: `DATA_W] = scores[gh][gs][gs2];
                end
                assign sm_in[gh][gs] = temp_sm_in;
            end
        end
    endgenerate

    // 5. Unpack Softmax output and multiply by V
    always @(*) begin
        for (h = 0; h < `NUM_HEADS; h = h + 1) begin
            for (s = 0; s < `SEQ_LEN; s = s + 1) begin
                for (s2 = 0; s2 < `SEQ_LEN; s2 = s2 + 1) begin
                    probs[h][s][s2] = sm_out[h][s][s2*`DATA_W +: `DATA_W];
                end
                
                for (d = 0; d < `HEAD_DIM; d = d + 1) begin
                    acc = 0;
                    for (s2 = 0; s2 < `SEQ_LEN; s2 = s2 + 1) begin
                        acc = acc + (probs[h][s][s2] * V[h][s2][d]);
                    end
                    head_out[h][s][d] = acc >>> `FRAC_W;
                end
            end
        end

        // 6. Concatenate heads
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (h = 0; h < `NUM_HEADS; h = h + 1) begin
                for (d = 0; d < `HEAD_DIM; d = d + 1) begin
                    concat_out[s][h*`HEAD_DIM + d] = head_out[h][s][d];
                end
            end
        end

        // 7. Final Projection (O = concat_out * Wo)
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                acc = 0;
                for (j = 0; j < `EMBED_DIM; j = j + 1) begin
                    acc = acc + (concat_out[s][j] * Wo[j][i]);
                end
                out[(s*`EMBED_DIM + i)*`DATA_W +: `DATA_W] = acc >>> `FRAC_W;
            end
        end
    end
endmodule
