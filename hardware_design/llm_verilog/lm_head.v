`timescale 1ns / 1ps
`include "llm_params.vh"

module lm_head (
    input wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] in_vec,
    output reg [`SEQ_LEN*8-1:0] predicted_tokens // 8 bits per token
);

    reg signed [`DATA_W-1:0] H [0:`SEQ_LEN-1][0:`EMBED_DIM-1];
    reg signed [`DATA_W-1:0] logits [0:`SEQ_LEN-1][0:`VOCAB_SIZE-1];
    
    // LM Head Weights (EMBED_DIM x VOCAB_SIZE)
    reg signed [`DATA_W-1:0] W_lm [0:`EMBED_DIM-1][0:`VOCAB_SIZE-1];
    
    integer i, j, s;
    initial begin
        for (i = 0; i < `EMBED_DIM; i = i + 1) begin
            for (j = 0; j < `VOCAB_SIZE; j = j + 1) begin
                // Simple deterministic initialization
                W_lm[i][j] = (i + j) << 2; // Scaled to Q8.8
            end
        end
    end

    reg signed [31:0] acc;
    reg signed [`DATA_W-1:0] max_logit;
    reg [7:0] best_token;

    always @(*) begin
        // 1. Unpack input
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                H[s][i] = in_vec[(s*`EMBED_DIM + i)*`DATA_W +: `DATA_W];
            end
        end

        // 2. Linear Projection to Logits
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            max_logit = -32768; // Min signed 16-bit
            best_token = 8'd0;
            
            for (j = 0; j < `VOCAB_SIZE; j = j + 1) begin
                acc = 0;
                for (i = 0; i < `EMBED_DIM; i = i + 1) begin
                    acc = acc + (H[s][i] * W_lm[i][j]);
                end
                logits[s][j] = acc >>> `FRAC_W;
                
                // 3. Argmax Decoding (find token with highest logit)
                if (logits[s][j] > max_logit) begin
                    max_logit = logits[s][j];
                    best_token = j[7:0];
                end
            end
            
            predicted_tokens[s*8 +: 8] = best_token;
        end
    end

endmodule
