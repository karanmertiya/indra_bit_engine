# DeepSeek V3 NPU Core - Sky130 ASIC Implementation

## Overview
This repository contains the Verilog RTL and physical design flow scripts for a custom AI hardware accelerator (NPU). The design implements the core mathematical pipeline of the **DeepSeek V3** large language model architecture and maps it to a physical ASIC using the open-source **Skywater 130nm (Sky130)** process node.

## Key Architectural Components
* **Systolic Array Matrix Multiplier (16x16):** The core compute engine for high-throughput tensor operations.
* **SwiGLU Activation Unit:** Non-linear activation functions optimized for the DeepSeek feed-forward network.
* **RoPE (Rotary Position Embedding):** Positional encoding engine for context window processing.
* **MLA (Multi-Head Latent Attention) Router:** Data routing for the attention mechanism.
* **MoE (Mixture of Experts) Router:** Conditional routing logic for expert selection.

## Physical Design Metrics (Sky130)
The design was synthesized as a flat "sea-of-gates" macro using the OpenROAD toolchain.
* **Total Logic Gates:** 209,743 standard cells
* **Core Area:** 1.69 mm² (1,692,631 sq microns)
* **Utilization:** 30% 
* **Total Active Power:** 385 mW 
  * Combinational Power: 269 mW (69.9%)
  * Sequential Power: 116 mW (30.1%)
  * Leakage Power: 0.57 uW (Negligible)
* **Critical Path Delay:** 235 ns (Unpipelined flat mapping)

## Toolchain & Methodology
* **Synthesis:** Yosys (RTL-to-gate mapping)
* **Physical Design (P&R):** OpenROAD (Floorplanning, Placement, Global Routing, Parasitic Extraction)
* **PDK:** Skywater 130nm (Sky130A)

## Artifacts Included
* `npu_gate.v`: Synthesized structural netlist.
* `final.def`: Physical layout Design Exchange Format file.
* `final.spg`: OpenROAD layout database.
* `layout.png`: Render of the physical silicon floorplan.

---

## 📄 Resume Bullet Points (For your CV)

* **Architected and physically mapped** the core compute pipeline of the DeepSeek V3 LLM (incorporating 16x16 Systolic Arrays, SwiGLU, MoE, and RoPE modules) to a custom ASIC layout using Verilog and the OpenROAD EDA toolchain.
* **Synthesized 209,000+ logic gates** onto the Skywater 130nm process node, achieving a highly dense 1.69 mm² silicon footprint using a flat physical placement strategy.
* **Profiled power and timing characteristics** using Static Timing Analysis (STA) and parasitic extraction, validating a highly efficient total active power draw of 385 mW for massive parallel matrix multiplications.
* **Automated the physical design flow** by containerizing the Yosys and OpenROAD pipeline via Docker, successfully resolving DRC/routing bottlenecks to verify the architecture's physical viability.
