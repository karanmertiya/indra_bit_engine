`timescale 1ns / 1ps
`include "llm_params.vh"

module transformer_block (
    input wire clk,
    input wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] in_vec,
    output reg [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] out_vec
);

    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] norm1_out;
    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] attn_out;
    reg [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] residual_add1;
    
    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] norm2_out;
    wire [`SEQ_LEN*`EMBED_DIM*`DATA_W-1:0] ffn_out;

    genvar i;
    
    // 1. RMSNorm 1 (Pre-Attention)
    // Instantiating one rmsnorm per sequence token
    generate
        for (i = 0; i < `SEQ_LEN; i = i + 1) begin : gen_norm1
            rmsnorm u_norm1 (
                .in_vec(in_vec[i*`EMBED_DIM*`DATA_W +: `EMBED_DIM*`DATA_W]),
                .out_vec(norm1_out[i*`EMBED_DIM*`DATA_W +: `EMBED_DIM*`DATA_W])
            );
        end
    endgenerate

    // 2. Multi-Head Attention
    attention u_attn (
        .clk(clk),
        .x_in(norm1_out),
        .out(attn_out)
    );

    // 3. Residual Connection 1
    integer s, d;
    always @(*) begin
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (d = 0; d < `EMBED_DIM; d = d + 1) begin
                automatic reg signed [`DATA_W-1:0] val_in = in_vec[(s*`EMBED_DIM + d)*`DATA_W +: `DATA_W];
                automatic reg signed [`DATA_W-1:0] val_attn = attn_out[(s*`EMBED_DIM + d)*`DATA_W +: `DATA_W];
                residual_add1[(s*`EMBED_DIM + d)*`DATA_W +: `DATA_W] = val_in + val_attn;
            end
        end
    end

    // 4. RMSNorm 2 (Pre-FFN)
    generate
        for (i = 0; i < `SEQ_LEN; i = i + 1) begin : gen_norm2
            rmsnorm u_norm2 (
                .in_vec(residual_add1[i*`EMBED_DIM*`DATA_W +: `EMBED_DIM*`DATA_W]),
                .out_vec(norm2_out[i*`EMBED_DIM*`DATA_W +: `EMBED_DIM*`DATA_W])
            );
        end
    endgenerate

    // 5. Feed Forward Network
    ffn u_ffn (
        .in_vec(norm2_out),
        .out_vec(ffn_out)
    );

    // 6. Residual Connection 2
    always @(*) begin
        for (s = 0; s < `SEQ_LEN; s = s + 1) begin
            for (d = 0; d < `EMBED_DIM; d = d + 1) begin
                automatic reg signed [`DATA_W-1:0] val_res1 = residual_add1[(s*`EMBED_DIM + d)*`DATA_W +: `DATA_W];
                automatic reg signed [`DATA_W-1:0] val_ffn = ffn_out[(s*`EMBED_DIM + d)*`DATA_W +: `DATA_W];
                out_vec[(s*`EMBED_DIM + d)*`DATA_W +: `DATA_W] = val_res1 + val_ffn;
            end
        end
    end

endmodule
