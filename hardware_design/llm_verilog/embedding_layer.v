`timescale 1ns / 1ps
`include "llm_params.vh"

module embedding_layer (
    input wire clk,
    input wire [7:0] token_id,      // Vocab size 256
    input wire [4:0] pos_id,        // Sequence length up to 32
    output reg [(`EMBED_DIM * `DATA_W) - 1 : 0] embedded_out
);

    // Simulated ROM for Token Embeddings (256 x 32 x 16-bit)
    // In a real ASIC, this would be an SRAM macro or ROM compiler output.
    reg [`DATA_W-1:0] token_rom [0:`VOCAB_SIZE-1][0:`EMBED_DIM-1];
    
    // Simulated ROM for Positional Embeddings (32 x 32 x 16-bit)
    reg [`DATA_W-1:0] pos_rom [0:`SEQ_LEN-1][0:`EMBED_DIM-1];

    // Initialization with pseudo-deterministic values for simulation
    integer i, j;
    initial begin
        for (i = 0; i < `VOCAB_SIZE; i = i + 1) begin
            for (j = 0; j < `EMBED_DIM; j = j + 1) begin
                // Example: token_id + dimension_index as a Q8.8 value
                token_rom[i][j] = (i + j) << 2; 
            end
        end
        for (i = 0; i < `SEQ_LEN; i = i + 1) begin
            for (j = 0; j < `EMBED_DIM; j = j + 1) begin
                // Sinusoidal-like deterministic value
                pos_rom[i][j] = (i * j) << 4; 
            end
        end
    end

    // Combinational read and addition (Token + Positional)
    integer k;
    always @(*) begin
        for (k = 0; k < `EMBED_DIM; k = k + 1) begin
            // embedded_out = token_embed + pos_embed
            embedded_out[k * `DATA_W +: `DATA_W] = token_rom[token_id][k] + pos_rom[pos_id][k];
        end
    end

endmodule
