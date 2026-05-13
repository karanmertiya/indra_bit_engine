import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void apot4_snap_kernel(float* in_data, float* out_data, int numel, float min_s_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float val = in_data[i];
        if (val == 0.0f) { out_data[i] = 0.0f; return; }
        
        float mag = fmaxf(fabsf(val), min_s_val);
        float sign_val = val > 0 ? 1.0f : -1.0f;

        float k1 = roundf(log2f(mag));
        float p1 = exp2f(k1);
        float res1 = mag - p1;
        float k2 = fminf(roundf(log2f(fmaxf(fabsf(res1), min_s_val))), k1 - 1.0f);
        float p2 = (res1 > 0 ? 1.0f : -1.0f) * exp2f(k2);
        float res2 = res1 - p2;
        float k3 = fminf(roundf(log2f(fmaxf(fabsf(res2), min_s_val))), k2 - 1.0f);
        float p3 = (res2 > 0 ? 1.0f : -1.0f) * exp2f(k3);
        float res3 = res2 - p3;
        float k4 = fminf(roundf(log2f(fmaxf(fabsf(res3), min_s_val))), k3 - 1.0f);
        float p4 = (res3 > 0 ? 1.0f : -1.0f) * exp2f(k4);
        
        out_data[i] = sign_val * (p1 + p2 + p3 + p4);
    }
}

torch::Tensor apot4_snap_cuda(torch::Tensor x, float min_s) {
    auto result = torch::empty_like(x);
    int numel = x.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    float min_s_val = exp2f(min_s);
    apot4_snap_kernel<<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), numel, min_s_val);
    return result;
}
"""

_bit_cuda = None

def get_cuda_engine():
    global _bit_cuda
    if _bit_cuda is None:
        print("[Indra-Bit] Compiling C++ CUDA 4-Term Engine...")
        _bit_cuda = load_inline(
            name='bit_cuda', 
            cpp_sources="torch::Tensor apot4_snap_cuda(torch::Tensor x, float min_s);", 
            cuda_sources=cuda_source, 
            functions=['apot4_snap_cuda'], 
            with_cuda=True, 
            extra_cflags=['-O3'], 
            extra_cuda_cflags=['-O3']
        )
    return _bit_cuda

class CudaTurboOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.005, b1=0.9, b2=0.999, min_s=-20):
        super().__init__(params, dict(lr=lr, b1=b1, b2=b2, min_s=min_s))
        self.engine = get_cuda_engine()

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, ms = g['lr'], g['min_s']
            for p in g['params']:
                if p.grad is None: continue
                st = self.state[p]
                if not st: 
                    st['m'] = torch.zeros_like(p)
                    st['v'] = torch.zeros_like(p)
                st['m'].mul_(g['b1']).add_(p.grad, alpha=1-g['b1'])
                st['v'].mul_(g['b2']).addcmul_(p.grad, p.grad, value=1-g['b2'])
                U = (st['m'] / (st['v'].sqrt() + 1e-8)) * lr

                def bs(v):
                    lv = torch.log2(v.abs().clamp(min=2.0**(ms-5)))
                    k  = lv.floor() + (torch.rand_like(lv) < (lv - lv.floor())).float()
                    return v.sign() * 2.0**k.clamp(ms, 0)

                p.data.sub_(bs(U))
                if p.dim() > 1: 
                    p.data.copy_(self.engine.apot4_snap_cuda(p.data, float(ms)))
