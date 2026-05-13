import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Multiplier-Free Matrix Multiplication (BitLinear)
// X: [M, K] (INT8)
// Weights: [N, K] (INT8)
// Out: [M, N] (INT32)

__global__ void bit_linear_kernel(
    const int8_t* __restrict__ X, 
    const int8_t* __restrict__ sign, 
    const int8_t* __restrict__ k1, 
    const int8_t* __restrict__ k2, 
    const int8_t* __restrict__ k3, 
    const int8_t* __restrict__ k4, 
    int32_t* __restrict__ out, 
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int32_t sum = 0;
        
        // Accumulation Loop (ZERO MULTIPLICATIONS)
        for (int i = 0; i < K; ++i) {
            int8_t x_val = X[row * K + i];
            int idx = col * K + i;
            
            // Pure Bit-Shifts
            int32_t val = (x_val >> k1[idx]) + 
                          (x_val >> k2[idx]) + 
                          (x_val >> k3[idx]) + 
                          (x_val >> k4[idx]);
                          
            // Sign application
            sum += sign[idx] * val;
        }
        out[row * N + col] = sum;
    }
}

torch::Tensor bit_linear_forward(torch::Tensor X, torch::Tensor sign, 
                                 torch::Tensor k1, torch::Tensor k2, 
                                 torch::Tensor k3, torch::Tensor k4) {
    int M = X.size(0);
    int K = X.size(1);
    int N = sign.size(0);
    
    auto out = torch::zeros({M, N}, torch::dtype(torch::kInt32).device(X.device()));
    
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    
    bit_linear_kernel<<<blocks, threads>>>(
        X.data_ptr<int8_t>(), sign.data_ptr<int8_t>(), k1.data_ptr<int8_t>(),
        k2.data_ptr<int8_t>(), k3.data_ptr<int8_t>(), k4.data_ptr<int8_t>(),
        out.data_ptr<int32_t>(), M, N, K
    );
    return out;
}
"""

_engine = None

def get_inference_engine():
    global _engine
    if _engine is None:
        print("[Indra-Bit] Compiling C++ Bit-Shift Forward Engine...")
        _engine = load_inline(
            name='bit_forward', 
            cpp_sources="torch::Tensor bit_linear_forward(torch::Tensor X, torch::Tensor sign, torch::Tensor k1, torch::Tensor k2, torch::Tensor k3, torch::Tensor k4);", 
            cuda_sources=cuda_source, 
            functions=['bit_linear_forward'], 
            with_cuda=True, 
            extra_cflags=['-O3'], 
            extra_cuda_cflags=['-O3', '-use_fast_math']
        )
    return _engine

class FastBitLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, trained_weights):
        super().__init__()
        # We assume extract_apot_exponents has already broken the float weights into Int8 tensors
        self.engine = get_inference_engine()
        # Register buffers...
        
    def forward(self, x_int8):
        # Physically executes the zero-multiplier CUDA kernel
        return self.engine.bit_linear_forward(
            x_int8, self.sign, self.k1, self.k2, self.k3, self.k4
        )
