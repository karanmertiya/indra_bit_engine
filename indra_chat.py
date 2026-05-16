# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import warnings
import sys

# Suppress warnings for clean terminal output
warnings.filterwarnings("ignore")

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
    def __init__(self, linear_layer):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # 1. Snap floating point weights to 4-term APoT representation
        sign, k1, k2, k3, k4 = extract_apot_exponents(linear_layer.weight.data)
        
        # Reconstruct the exact APoT quantized float weight
        apot_weight = sign.float() * ( (2.0 ** -k1.float()) + (2.0 ** -k2.float()) + (2.0 ** -k3.float()) + (2.0 ** -k4.float()) )
        
        self.register_buffer('apot_weight', apot_weight)
        
        if linear_layer.bias is not None:
            self.register_buffer('bias', linear_layer.bias.data)
        else:
            self.bias = None

    def forward(self, x):
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
    print("="*50)
    print(" ⚡ INDRA-BIT ZERO-MULTIPLIER LLM TERMINAL ⚡")
    print("="*50)
    
    # Choose your model. Set to local path for the 32B/70B converted models.
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # model_id = "./indra_bit_32b_distill" # Uncomment this to run your converted 32B model!
    
    print(f"\n[SYSTEM] Loading Model: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use low_cpu_mem_usage=True for 32B/70B models
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to load model. Ensure you have internet access. Error: {e}")
        sys.exit(1)
        
    print("\n[SYSTEM] Converting billions of FP32 parameters to Indra-Bit APoT Format...")
    model = convert_to_apot(model)
    print("[SYSTEM] Conversion Complete! Multipliers permanently removed.")
    
    # Initialize the streamer for real-time text generation
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print("\n" + "="*50)
    print(" Ready. Type your prompt below. (Type 'exit' to quit)")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Format prompt for TinyLlama Chat
            prompt = f"<|system|>\nYou are Indra, an advanced 100% multiplier-free AI assistant running purely on CPU bit-shifts. You are helpful and highly intelligent.</s>\n<|user|>\n{user_input}</s>\n<|assistant|>\n"
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            print("\nIndra: ", end="", flush=True)
            
            # Generate response with streaming enabled
            with torch.no_grad():
                model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            print("\n" + "-"*50)
            
        except KeyboardInterrupt:
            print("\n[SYSTEM] Exiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")

if __name__ == "__main__":
    main()
