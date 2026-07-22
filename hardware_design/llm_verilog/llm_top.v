`timescale 1ns / 1ps
`include "llm_params.vh"

module llm_top (
    input wire clk,
    input wire rst,
    input wire [`SEQ_LEN*8-1:0] input_tokens,
    output wire [`SEQ_LEN*8-1:0] predicted_tokens
);

    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] embedded_seq;
    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] block1_out;
    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] block2_out;
    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] final_norm_out;

    genvar i;
    
    // 1. Token & Positional Embeddings
    generate
        for (i = 0; i < `SEQ_LEN; i = i + 1) begin : gen_embed
            embedding_layer u_embed (
                .clk(clk),
                .token_id(input_tokens[i*8 +: 8]),
                .pos_id(i[4:0]),
                .embedded_out(embedded_seq[i*`EMBED_DIM*`DATA_W +: `EMBED_DIM*`DATA_W])
            );
        end
    endgenerate

    // 2. Transformer Block 1
    transformer_block u_block1 (
        .clk(clk),
        .in_vec(embedded_seq),
        .out_vec(block1_out)
    );

    // 3. Transformer Block 2
    transformer_block u_block2 (
        .clk(clk),
        .in_vec(block1_out),
        .out_vec(block2_out)
    );

    // 4. Final RMSNorm
    generate
        for (i = 0; i < `SEQ_LEN; i = i + 1) begin : gen_final_norm
            rmsnorm u_final_norm (
                .in_vec(block2_out[i*`EMBED_DIM*`DATA_W +: `EMBED_DIM*`DATA_W]),
                .out_vec(final_norm_out[i*`EMBED_DIM*`DATA_W +: `EMBED_DIM*`DATA_W])
            );
        end
    endgenerate

    // 5. LM Head & Argmax Decoding
    lm_head u_lm_head (
        .in_vec(final_norm_out),
        .predicted_tokens(predicted_tokens)
    );

endmodule
