`timescale 1ns / 1ps
`include "llm_params.vh"

module ffn (
    input wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] in_vec,
    output reg [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] out_vec
);

    reg signed [`DATA_W-1:0] X [0:`SEQ_LEN-1][0:`EMBED_DIM-1];
    reg signed [`DATA_W-1:0] H [0:`SEQ_LEN-1][0:`FFN_DIM-1];
    reg signed [`DATA_W-1:0] O [0:`SEQ_LEN-1][0:`EMBED_DIM-1];

    // Simulated ROM Weights
    reg signed [`DATA_W-1:0] W1 [0:`EMBED_DIM-1][0:`FFN_DIM-1];
    reg signed [`DATA_W-1:0] W2 [0:`FFN_DIM-1][0:`EMBED_DIM-1];
    
    integer i, j, s;
    initial begin
        // Initialize weights (identity-like or small constant for simulation)
        for (i = 0; i < `EMBED_DIM; i = i + 1) begin
            for (j = 0; j < `FFN_DIM; j = j + 1) begin
                W1[i][j] = (i == j) ? 16'h0100 : 16'h0000;
            end
        end
        for (i = 0; i < `FFN_DIM; i = i + 1) begin
            for (j = 0; j < `EMBED_DIM; j = j + 1) begin
                W2[i][j] = (i == j) ? 16'h0100 : 16'h0000;
            end
        end
    end

    reg signed [31:0] acc;

    always @(*) begin
        // 1. Unpack input
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                X[s][i] = in_vec[(s*`EMBED_DIM + i)*`DATA_W +: `DATA_W];
            end
        end

        // 2. Linear 1 (X * W1) + ReLU
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (j = 0; j < `FFN_DIM; j = j + 1) begin
                acc = 0;
                for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                    acc = acc + (X[s][i] * W1[i][j]);
                end
                acc = acc >>> `FRAC_W;
                
                // Hardware ReLU: if negative (sign bit 15), set to 0
                if (acc[15] == 1'b1) begin
                    H[s][j] = 16'h0000;
                end else begin
                    H[s][j] = acc[15:0];
                end
            end
        end

        // 3. Linear 2 (H * W2)
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (j = 0; j < `EMBED_DIM; j = j + 1) begin
                acc = 0;
                for (i = 0; i < `FFN_DIM; i = i + 1) begin
                    acc = acc + (H[s][i] * W2[i][j]);
                end
                O[s][j] = acc >>> `FRAC_W;
                
                // Pack output
                out_vec[(s*`EMBED_DIM + j)*`DATA_W +: `DATA_W] = O[s][j];
            end
        end
    end

endmodule
