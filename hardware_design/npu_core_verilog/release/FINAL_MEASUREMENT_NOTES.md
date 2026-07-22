# Final Measurement Notes

This release verifies a small LLM inference tile, not a full DeepSeek-V3 model.

## What is verified

- Two complete FSM runs in Icarus Verilog.
- Raw Q projection checked against Python golden output.
- Raw K projection checked against Python golden output.
- Q_RoPE checked against Python golden output.
- 149 RTL clock cycles per run in the provided testbench.
- Testbench clock is 100 MHz, so the simulated run time is 1.49 us at that assumed clock.
- Yosys generic synthesis/check reports 0 structural problems for npu_tile_top.

## What is not verified

- Full DeepSeek-V3 end-to-end inference.
- Attention softmax, V projection, MLA cache update, MoE expert compute, SwiGLU MLP, logits, sampling, or tokenizer.
- Physical ASIC area, timing, power, yield, mask cost, packaging cost, or dollars per token.
- GPU/CPU speedup against a fair benchmark.
- Real bandwidth counters: bytes_read and bytes_written are placeholder ports and must not be used as measured bandwidth.

## Defensible calculations

- Array size: 16 x 16 = 256 MAC cells.
- If every MAC were busy every cycle at the testbench clock of 100 MHz, peak MAC rate is 25.6 GMAC/s.
- If counting multiply and add separately, that is 51.2 GOPS peak arithmetic throughput.
- The verified workload completes in 149 cycles, or 1.49 us at 100 MHz.

These are RTL/testbench-derived values, not silicon measurements.
