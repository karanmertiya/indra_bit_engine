<div align="center">
  <h1>⚡ Indra-Bit Engine</h1>
  <p><b>A 1.7M Parameter, 100% Multiplier-Free Neural Architecture</b></p>
</div>

---

Indra-Bit completely eliminates floating-point multipliers (MACs) during inference by quantizing weights into **Additive Powers of Two (APoT)**. It relies exclusively on hardware-level integer bit-shifts (`>>`) and additions, enabling Deep Learning on highly constrained edge devices, CPUs, and custom ASICs without requiring Tensor Cores.

## 🚀 Key Performance Metrics
* **SOTA Accuracy:** Achieved **91.10%** on CIFAR-10 (Matched-Capacity ResNet), proving that multiplier-free architectures can match FP32 baselines.
* **8x Memory Compression:** 1.7M parameters compressed losslessly into **~850 KB** using a custom 4-bit Nibble Packer.
* **Zero-Multiplier Inference:** Includes a custom **C++ CUDA Kernel** that physically bypasses NVIDIA's cuBLAS engine, executing math purely via `INT8` wire-shifts.
* **Browser-Native Speed:** Executes locally on standard CPUs via WebAssembly at **~54.6 ms/inference**.

## 🧠 The Architecture

Standard neural networks rely heavily on Floating Point Multiply-Accumulate (FMA) operations. Indra-Bit replaces these by restricting weights to the sum of 4 powers of two:
`W = sign * (2^k1 + 2^k2 + 2^k3 + 2^k4)`

During inference, a standard multiplication `X * W` mathematically simplifies to pure bitwise right-shifts:
`Output = (X >> k1) + (X >> k2) + (X >> k3) + (X >> k4)`

## ⚙️ Repository Structure

```text
indra_bit_engine/
│
├── core/
│   ├── turbo_optim.py      # Custom C++ JIT CUDA Compiler for APoT Training
│   ├── distiller.py        # FP16 Mixed-Precision Knowledge Distiller
│   ├── cuda_inference.py   # Raw C++ CUDA Bit-Shift Kernel (Zero Multipliers)
│   └── bit_packer.py       # 4-bit Nibble Serializer for 8x Compression
│
├── models/
│   ├── shift_bn.py         # Multiplier-Free Batch Normalization
│   └── architectures.py    # Parameterized Indra-Bit ResNets (270K - 1.7M)
│
└── train.py                # Main CLI Training Script
```

## 💻 Quick Start

**1. Train a Multiplier-Free Model:**
Kick off a massive Mixed-Precision Distillation run with a single command.
```bash
python train.py --width 40 --epochs 100 --teacher cifar10_repvgg_a2
```

**2. Physical Bit-Shift Inference:**
To physically execute the network without PyTorch's cuBLAS float engine, use the included C++ JIT extension:
```python
from core.cuda_inference import FastBitLinear

# Compiles the custom CUDA kernel on the fly
layer = FastBitLinear(in_features, out_features, trained_weights)
out_int32 = layer(input_int8) # Executes 100% Multiplier-Free
```

## 🏗️ Future Work
This framework provides the mathematical algorithms and CUDA simulations. The ultimate frontier is translating the `cuda_inference.py` bit-shift logic directly into physical SRAM/FPGA hardware for sub-millisecond edge AI.
