import numpy as np
import time
import os

# --- Q16.16 Fixed Point Helpers ---
def to_q16(val):
    return np.int32(np.round(val * 65536.0))

def from_q16(val):
    return val / 65536.0

# --- INT8 Helpers ---
def to_int8(val):
    return np.clip(np.round(val), -128, 127).astype(np.int8)

# --- Golden Model ---
class EndToEndInferenceTile:
    def __init__(self, hidden_dim=16):
        self.d = hidden_dim
        # Random FP32 weights (simulating trained model)
        np.random.seed(42)
        
        # Dense Projections (INT8 weights)
        self.W_q = to_int8(np.random.uniform(-10, 10, (self.d, self.d)))
        self.W_k = to_int8(np.random.uniform(-10, 10, (self.d, self.d)))
        
        # MoE Centroids (Q16.16)
        self.experts = 4
        self.expert_centroids = to_q16(np.random.uniform(-1, 1, (self.experts, self.d)))
        
        # SwiGLU MLP Experts (INT8)
        self.W_gate = to_int8(np.random.uniform(-5, 5, (self.experts, self.d, self.d)))
        self.W_up   = to_int8(np.random.uniform(-5, 5, (self.experts, self.d, self.d)))
        self.W_down = to_int8(np.random.uniform(-5, 5, (self.experts, self.d, self.d)))

    def hard_swish_q16(self, x_q16):
        # Q16.16 Hard Swish: x * clip((x + 3.0)/6.0, 0, 1)
        three = to_q16(3.0)
        six = to_q16(6.0)
        
        # Add 3
        x_plus_3 = x_q16 + three
        
        # Divide by 6 (Shift approximation or exact div)
        div_6 = (np.int64(x_plus_3) << 16) // six
        
        # Clip [0, 1]
        clipped = np.clip(div_6, 0, 65536) # 1.0 in Q16.16 is 65536
        
        # Multiply x * clipped
        result = (np.int64(x_q16) * clipped) >> 16
        return np.int32(result)

    def rope_q16(self, x_q16, pos=0):
        # Extremely simplified RoPE mapping (pairs of 2)
        # Using precomputed sine/cosine for pos 0
        freq = 10000.0 ** (-0 / (self.d))
        theta = pos * freq
        cos_t = to_q16(np.cos(theta))
        sin_t = to_q16(np.sin(theta))
        
        out = np.zeros_like(x_q16)
        for i in range(0, self.d, 2):
            x0 = x_q16[i]
            x1 = x_q16[i+1]
            out[i]   = ((np.int64(x0) * cos_t) - (np.int64(x1) * sin_t)) >> 16
            out[i+1] = ((np.int64(x0) * sin_t) + (np.int64(x1) * cos_t)) >> 16
        return np.int32(out)

    def forward(self, x_int8):
        start_time = time.perf_counter()
        
        # 1. Q, K Projections (Systolic Array math: INT8 * INT8 = INT32)
        q_int32 = np.dot(x_int8.astype(np.int32), self.W_q.astype(np.int32))
        k_int32 = np.dot(x_int8.astype(np.int32), self.W_k.astype(np.int32))
        
        # Convert to Q16.16 for RoPE
        q_q16 = q_int32.astype(np.int32)
        k_q16 = k_int32.astype(np.int32)
        
        q_rope = np.zeros_like(q_q16)
        for row in range(16):
            q_rope[row] = self.rope_q16(q_q16[row], pos=row)
            
        # 3. MoE Router (Dot product between Q_rope and Centroids)
        # Top-1 Selection for each token (in reality we route per token, for simplified benchmark we pick expert for token 0)
        scores = np.zeros(self.experts, dtype=np.int64)
        for e in range(self.experts):
            scores[e] = np.dot(np.int64(q_rope[0]), np.int64(self.expert_centroids[e])) >> 16
        
        best_expert = np.argmax(scores)
        
        # 4. SwiGLU MLP (for the selected expert)
        mlp_in_int8 = to_int8(from_q16(q_rope))
        
        gate_int32 = np.dot(mlp_in_int8.astype(np.int32), self.W_gate[best_expert].astype(np.int32))
        up_int32   = np.dot(mlp_in_int8.astype(np.int32), self.W_up[best_expert].astype(np.int32))
        
        gate_q16 = gate_int32.astype(np.int32)
        up_q16   = up_int32.astype(np.int32)
        
        swish_up_q16 = np.zeros_like(up_q16)
        for row in range(16):
            for i in range(self.d):
                swish_up_q16[row][i] = self.hard_swish_q16(up_q16[row][i])
            
        activated_q16 = (np.int64(gate_q16) * np.int64(swish_up_q16)) >> 16
        
        # Down Projection
        activated_int8 = to_int8(from_q16(activated_q16))
        down_int32 = np.dot(activated_int8.astype(np.int32), self.W_down[best_expert].astype(np.int32))
        
        end_time = time.perf_counter()
        
        return down_int32, best_expert, (end_time - start_time)

    def generate_hex_files(self):
        # Generate random input tokens (16 tokens of 16 dims)
        x_int8 = to_int8(np.random.uniform(-10, 10, (16, self.d)))
        
        # Run forward pass
        out_int32, best_expert, latency = self.forward(x_int8)
        
        print(f"Software Inference Latency: {latency * 1e6:.2f} microseconds")
        print(f"Chosen MoE Expert: {best_expert}")
        
        # Dump input
        with open("tb_input.hex", "w") as f:
            for row in x_int8:
                for val in row:
                    f.write(f"{int(val) & 0xFF:02x}\n")
                
        # Dump output
        with open("tb_golden_out.hex", "w") as f:
            for row in out_int32:
                for val in row:
                    f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
                
        # Dump Weights
        with open("W_q.hex", "w") as f:
            for row in self.W_q:
                for val in row:
                    f.write(f"{int(val) & 0xFF:02x}\n")
        with open("W_k.hex", "w") as f:
            for row in self.W_k:
                for val in row:
                    f.write(f"{int(val) & 0xFF:02x}\n")
        with open("W_gate.hex", "w") as f:
            for e in range(self.experts):
                for val in self.W_gate[e].flatten():
                    f.write(f"{int(val) & 0xFF:02x}\n")
        with open("W_up.hex", "w") as f:
            for e in range(self.experts):
                for val in self.W_up[e].flatten():
                    f.write(f"{int(val) & 0xFF:02x}\n")
        with open("W_down.hex", "w") as f:
            for e in range(self.experts):
                for val in self.W_down[e].flatten():
                    f.write(f"{int(val) & 0xFF:02x}\n")
        with open("expert_centroids.hex", "w") as f:
            for row in self.expert_centroids:
                for val in row:
                    f.write(f"{int(val) & 0xFFFFFFFF:08x}\n")
                
        print("Exported all hex files")

if __name__ == "__main__":
    tile = EndToEndInferenceTile()
    tile.generate_hex_files()
