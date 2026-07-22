`ifndef LLM_PARAMS_VH
`define LLM_PARAMS_VH

// Micro-LLM Configuration Parameters
// Scale: Toy-sized model for ASIC synthesis and simulation

`define VOCAB_SIZE 256
`define SEQ_LEN 16         // Max tokens in sequence
`define EMBED_DIM 32       // Hidden size (d_model)
`define NUM_HEADS 4        // Number of attention heads
`define HEAD_DIM (`EMBED_DIM / `NUM_HEADS) // 8 bits per head
`define FFN_DIM (`EMBED_DIM * 4) // 128

// Data format: 16-bit Q8.8 Fixed Point
`define DATA_W 16
`define FRAC_W 8

`endif // LLM_PARAMS_VH
