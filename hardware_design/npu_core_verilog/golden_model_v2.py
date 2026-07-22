import numpy as np

def to_int8(val):
    return np.clip(np.round(val), -128, 127).astype(np.int8)

def to_q16(val):
    return np.int32(np.round(val * 65536.0))

class Phase1InferenceTile:
    def __init__(self, d=16):
        self.d = d
        np.random.seed(1337)
        self.W_q = to_int8(np.random.uniform(-10, 10, (self.d, self.d)))
        self.W_k = to_int8(np.random.uniform(-10, 10, (self.d, self.d)))
        
        # Hardcoded LUT values matching Verilog (Q16.16)
        self.sin_lut = [0x00000000, 0x00006203, 0x0000B504, 0x0000ED70,
                         0x00010000, 0x0000ED70, 0x0000B504, 0x00006203,
                         0x00000000, -0x00006203, -0x0000B504, -0x0000ED70,
                         -0x00010000, -0x0000ED70, -0x0000B504, -0x00006203]
        self.cos_lut = [0x00010000, 0x0000ED70, 0x0000B504, 0x00006203,
                         0x00000000, -0x00006203, -0x0000B504, -0x0000ED70,
                         -0x00010000, -0x0000ED70, -0x0000B504, -0x00006203,
                         0x00000000, 0x00006203, 0x0000B504, 0x0000ED70]

    def rope_q16(self, x_q16, pos=0):
        out = np.zeros_like(x_q16)
        for i in range(0, self.d, 2):
            x0 = x_q16[i]
            x1 = x_q16[i+1]
            
            # Hardware approximation: theta_idx = (pos + dim) & 4'hF
            theta_idx = (pos + i) & 15
            
            cos_t = self.cos_lut[theta_idx]
            sin_t = self.sin_lut[theta_idx]
            
            out[i]   = ((np.int64(x0) * cos_t) - (np.int64(x1) * sin_t)) >> 16
            out[i+1] = ((np.int64(x0) * sin_t) + (np.int64(x1) * cos_t)) >> 16
        return np.int32(out)

    def forward(self, x_int8):
        q_int32 = np.dot(x_int8.astype(np.int32), self.W_q.astype(np.int32))
        k_int32 = np.dot(x_int8.astype(np.int32), self.W_k.astype(np.int32))
        
        q_rope = np.zeros_like(q_int32)
        for row in range(16):
            q_rope[row] = self.rope_q16(q_int32[row], pos=row)
            
        return q_rope, q_int32, k_int32

    def generate_files(self):
        x_int8 = to_int8(np.random.uniform(-5, 5, (16, self.d)))
        q_rope, q_int32, k_int32 = self.forward(x_int8)

        with open("sram_act.hex", "w") as f_act, open("sram_wgt.hex", "w") as f_wgt:
            for cycle in range(31):
                act_line = []
                wgt_line = []
                for j in range(16):
                    if cycle >= j and cycle < 16 + j:
                        act_line.append(x_int8[j][cycle - j])
                    else:
                        act_line.append(0)
                        
                    if cycle >= j and cycle < 16 + j:
                        wgt_line.append(self.W_q[cycle - j][j])
                    else:
                        wgt_line.append(0)
                
                act_hex = "".join([f"{int(act_line[c]) & 0xFF:02x}" for c in reversed(range(16))])
                wgt_hex = "".join([f"{int(wgt_line[c]) & 0xFF:02x}" for c in reversed(range(16))])
                f_act.write(f"{act_hex}\n")
                f_wgt.write(f"{wgt_hex}\n")

            f_act.write("00000000000000000000000000000000\n")
            f_wgt.write("00000000000000000000000000000000\n")

            for cycle in range(31):
                act_line = []
                wgt_line = []
                for j in range(16):
                    if cycle >= j and cycle < 16 + j:
                        act_line.append(x_int8[j][cycle - j])
                    else:
                        act_line.append(0)
                        
                    if cycle >= j and cycle < 16 + j:
                        wgt_line.append(self.W_k[cycle - j][j])
                    else:
                        wgt_line.append(0)
                
                act_hex = "".join([f"{int(act_line[c]) & 0xFF:02x}" for c in reversed(range(16))])
                wgt_hex = "".join([f"{int(wgt_line[c]) & 0xFF:02x}" for c in reversed(range(16))])
                f_act.write(f"{act_hex}\n")
                f_wgt.write(f"{wgt_hex}\n")
                
            for _ in range(64 - 31*2 - 1):
                f_act.write("00000000000000000000000000000000\n") 
                f_wgt.write("00000000000000000000000000000000\n")
                
        with open("sram_golden_q_rope.hex", "w") as f:
            for r in range(16): # TOP DOWN
                line_hex = "".join([f"{int(q_rope[r][c]) & 0xFFFFFFFF:08x}" for c in reversed(range(16))])
                f.write(f"{line_hex}\n")

        with open("sram_golden_q.hex", "w") as f:
            for r in range(16):
                line_hex = "".join([f"{int(q_int32[r][c]) & 0xFFFFFFFF:08x}" for c in reversed(range(16))])
                f.write(f"{line_hex}\n")

        with open("sram_golden_k.hex", "w") as f:
            for r in range(16):
                line_hex = "".join([f"{int(k_int32[r][c]) & 0xFFFFFFFF:08x}" for c in reversed(range(16))])
                f.write(f"{line_hex}\n")

if __name__ == "__main__":
    t = Phase1InferenceTile()
    t.generate_files()
    print("Generated Fixed LUT SRAM hex files.")
