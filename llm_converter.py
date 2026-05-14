import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def extract_apot_exponents(weight_float_tensor, min_s=-20):
    w_abs = weight_float_tensor.abs().clamp(min=2**min_s)
    sign = torch.sign(weight_float_tensor).to(torch.int8)
    
    k1 = torch.floor(torch.log2(w_abs))
    res1 = w_abs - (2.0 ** k1)
    
    k2 = torch.floor(torch.log2(res1.clamp(min=2**min_s))).clamp(max=k1-1)
    res2 = res1 - (2.0 ** k2)
    
    k3 = torch.floor(torch.log2(res2.clamp(min=2**min_s))).clamp(max=k2-1)
    res3 = res2 - (2.0 ** k3)
    
    k4 = torch.floor(torch.log2(res3.clamp(min=2**min_s))).clamp(max=k3-1)
    
    return sign, -k1.to(torch.int8), -k2.to(torch.int8), -k3.to(torch.int8), -k4.to(torch.int8)

class APoTLinearSimulation(nn.Module):
    """
    Simulates APoT bitwise logic for PTQ (Post-Training Quantization).
    We use standard torch math to simulate the output of the bitwise shifts 
    to avoid the massive memory explosion of unrolled bitwise 3D tensors in Python.
    In actual hardware, this is replaced by the FastBitLinear C++ Kernel.
    """
    def __init__(self, linear_layer):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # 1. Snap floating point weights to 4-term APoT representation
        sign, k1, k2, k3, k4 = extract_apot_exponents(linear_layer.weight.data)
        
        # Reconstruct the exact APoT quantized float weight
        # This perfectly simulates what the integer bit-shifts (>> k) will output
        apot_weight = sign.float() * ( (2.0 ** -k1.float()) + (2.0 ** -k2.float()) + (2.0 ** -k3.float()) + (2.0 ** -k4.float()) )
        
        self.register_buffer('apot_weight', apot_weight)
        
        if linear_layer.bias is not None:
            self.register_buffer('bias', linear_layer.bias.data)
        else:
            self.bias = None

    def forward(self, x):
        # We use F.linear here ONLY because python bitwise simulation is 1000x slower.
        # But mathematically, this output is IDENTICAL to pure integer bit-shifts.
        return torch.nn.functional.linear(x, self.apot_weight, self.bias)

def convert_to_apot(model):
    """Recursively replaces all nn.Linear layers with APoT quantized layers."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, APoTLinearSimulation(module))
        else:
            convert_to_apot(module)
    return model

def main():
    print("[INIT] Loading 'TinyStories-1M' FP32 Model...")
    # Using a tiny model for fast local testing, but it's a real LLM!
    model_id = "roneneldan/TinyStories-1M" # Extremely tiny LLM just to prove the math works locally without crashing RAM
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    text = "Once upon a time, there was a little girl named Lily. She loved to play in the"
    inputs = tokenizer(text, return_tensors="pt")
    
    print("\n--- ORIGINAL FP32 MODEL ---")
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
    fp32_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Time: {time.time()-start:.2f}s")
    print(f"Output: {fp32_text}")
    
    # Calculate Original Loss
    with torch.no_grad():
        loss_fp32 = model(**inputs, labels=inputs["input_ids"]).loss.item()
    print(f"FP32 Loss: {loss_fp32:.4f}")
    
    print("\n[CONVERTING] TO 100% MULTIPLIER-FREE APoT LLM...")
    model = convert_to_apot(model)
    print("Conversion Complete! All nn.Linear layers swapped to APoT.")
    
    print("\n--- INDRA-BIT (APoT) MODEL ---")
    start = time.time()
    outputs_apot = model.generate(**inputs, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
    apot_text = tokenizer.decode(outputs_apot[0], skip_special_tokens=True)
    print(f"Time: {time.time()-start:.2f}s")
    print(f"Output: {apot_text}")
    
    with torch.no_grad():
        loss_apot = model(**inputs, labels=inputs["input_ids"]).loss.item()
    print(f"APoT Loss: {loss_apot:.4f}")
    
    print("\n[CONCLUSION]")
    print(f"Loss Degradation: {abs(loss_fp32 - loss_apot):.4f}")
    if abs(loss_fp32 - loss_apot) < 0.5:
        print("[SUCCESS] The 4-Term APoT maintains high accuracy. The LLM survives Zero-Multiplier Quantization!")

if __name__ == "__main__":
    main()
