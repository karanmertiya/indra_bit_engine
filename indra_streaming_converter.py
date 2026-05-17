"""
INDRA-BIT "RESUMABLE GHOST" CONVERTER (70B STABLE)
==================================================
Key features:
1. RESUMABLE: If Kaggle restarts, it skips already uploaded chunks.
2. DISK-SAFE: Deletes as it goes.
3. RAM-SAFE: Peak ~9GB for 70B layers.
"""

import torch
import os
import gc
import psutil
import shutil
from tqdm import tqdm
from huggingface_hub import (
    hf_hub_download, list_repo_files,
    HfApi, create_repo
)
from safetensors import safe_open
from safetensors.torch import save_file

def ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def status(msg):
    print(f"[{ram_gb():.1f}GB RAM] {msg}")

def extract_apot_exponents(W, min_s=-14):
    W = W.to(torch.float16)
    w_abs = W.abs().clamp(min=2**min_s)
    sign = torch.sign(W)
    k1 = torch.floor(torch.log2(w_abs))
    res1 = w_abs - (2.0 ** k1)
    k2 = torch.floor(torch.log2(res1.clamp(min=2**min_s))).clamp(max=k1-1)
    res2 = res1 - (2.0 ** k2)
    k3 = torch.floor(torch.log2(res2.clamp(min=2**min_s))).clamp(max=k2-1)
    res3 = res2 - (2.0 ** k3)
    k4 = torch.floor(torch.log2(res3.clamp(min=2**min_s))).clamp(max=k3-1)
    apot_W = sign * ( (2.0 ** k1) + (2.0 ** k2) + (2.0 ** k3) + (2.0 ** k4) )
    return apot_W

def convert_and_upload_resumable(source_model_id, target_repo_id, hf_token):
    api = HfApi(token=hf_token)
    print(f"\n[SETUP] Target Repo: https://huggingface.co/{target_repo_id}")
    create_repo(repo_id=target_repo_id, token=hf_token, exist_ok=True)
    
    # Check what's already on the Hub to enable RESUMING
    print("[SETUP] Checking for existing files on HF Hub to enable resume...")
    try:
        existing_files = list_repo_files(target_repo_id, token=hf_token)
    except:
        existing_files = []

    all_source_files = list_repo_files(source_model_id, token=hf_token)
    weight_files = sorted([f for f in all_source_files if f.endswith(".safetensors")])
    config_files = [f for f in all_source_files if not f.endswith(".safetensors") and not f.startswith(".")]

    # 1. Configs
    for f in config_files:
        if f in existing_files:
            print(f"  Skipping config (exists): {f}")
            continue
        local = hf_hub_download(repo_id=source_model_id, filename=f, token=hf_token)
        api.upload_file(path_or_fileobj=local, path_in_repo=f, repo_id=target_repo_id, token=hf_token)
        print(f"  Uploaded config: {f}")

    # 2. Weights (The long part)
    print(f"\n[PROCESS] Converting {len(weight_files)} chunks...")
    for i, filename in enumerate(weight_files):
        if filename in existing_files:
            print(f"  >>> Skipping {filename} (Already on Hub) ✅")
            continue
            
        status(f"Processing Chunk {i+1}/{len(weight_files)}: {filename}")
        
        # Download (DIRECTLY TO /tmp, NO CACHE)
        temp_in = hf_hub_download(
            repo_id=source_model_id, 
            filename=filename, 
            token=hf_token,
            local_dir="/tmp/hf_download",
            local_dir_use_symlinks=False
        )
        
        # Convert
        new_tensors = {}
        with safe_open(temp_in, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for tensor_name in tqdm(keys, desc=f"  Quantizing {filename}"):
                tensor = f.get_tensor(tensor_name).to(torch.float16)
                if "weight" in tensor_name and any(x in tensor_name for x in ["self_attn", "mlp", "layers"]):
                    tensor = extract_apot_exponents(tensor)
                new_tensors[tensor_name] = tensor
                del tensor
        
        # Save & Upload
        temp_out = f"/tmp/converted_{filename.replace('/', '_')}"
        save_file(new_tensors, temp_out)
        del new_tensors
        gc.collect()
        
        api.upload_file(path_or_fileobj=temp_out, path_in_repo=filename, repo_id=target_repo_id, token=hf_token)
        
        # Cleanup BOTH input and output immediately
        if os.path.exists(temp_in):
            os.remove(temp_in)
        if os.path.exists(temp_out):
            os.remove(temp_out)
            
        # Also wipe the local_dir to be safe
        shutil.rmtree("/tmp/hf_download", ignore_errors=True)
        
        gc.collect()
        status(f"  ✅ {filename} Done. Disk usage reset to zero.")

    print("\n[FINISH] 70B Conversion Successful!")

if __name__ == "__main__":
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    HF_USERNAME = "SoloHacker007" 
    
    # 70B SETTINGS
    SOURCE = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    TARGET = f"{HF_USERNAME}/DeepSeek-R1-70B-IndraBit-APoT"
    
    convert_and_upload_resumable(SOURCE, TARGET, HF_TOKEN)
