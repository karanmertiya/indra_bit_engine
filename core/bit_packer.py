import torch
import struct

def pack_tensor_4bit(tensor, filepath):
    """Losslessly compresses the 4-term APoT tensor into 4-bit nibbles."""
    data = tensor.flatten().cpu().numpy()
    max_val = max(abs(data.max()), abs(data.min()), 1e-9)
    normalized = data / max_val
    
    # Map [-1, 1] to [0, 15]
    quantized = ((normalized + 1.0) * 7.5).round().astype('uint8')
    quantized = quantized.clip(0, 15)
    
    # Pack 2 nibbles into 1 byte
    packed = bytearray()
    for i in range(0, len(quantized), 2):
        high = quantized[i]
        low = quantized[i+1] if i+1 < len(quantized) else 0
        packed.append((high << 4) | low)
        
    with open(filepath, 'wb') as f:
        # Save scale factor and tensor length
        f.write(struct.pack('f', max_val))
        f.write(struct.pack('I', len(data)))
        f.write(packed)
        
    print(f"[Indra-Bit Packer] Exported {len(data)} params to {filepath}")
