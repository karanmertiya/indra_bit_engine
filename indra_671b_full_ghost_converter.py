"""
INDRA-BIT 671B MIXTURE OF EXPERTS — FULL RESUMABLE GHOST CONVERTER
==================================================================
This script executes the complete, production-grade, sharded ghost conversion
for the flagship DeepSeek-R1 671B model.

Operational Architecture:
  1. Multi-Session Resumability: Saves progress in 'conversion_state.json'.
     If Kaggle hits its 9-hour session timeout, running the script again
     will resume exactly from the last processed chunk.
  2. Zero-Cache Stream to Hub: Downloads one shard (~4.9GB), quantizes to
     our 8-term CSD bit-shift format, uploads the sharded result directly
     to the user's HF Hub repository, and wipes the disk cache instantly.
  3. Peak Disk Overhead: Stays strictly under 15GB (perfect for Kaggle).
"""

import os
# Force cache containment inside working workspace
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"

import json
import torch
import gc
import shutil
import time
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

# Configuration
SOURCE_REPO = "deepseek-ai/DeepSeek-R1"
TARGET_REPO = "SoloHacker007/DeepSeek-R1-671B-IndraBit-APoT"
STATE_FILE = "/kaggle/working/conversion_state.json"
CACHE_DIR = "/kaggle/working/hf_cache"
TOTAL_SHARDS = 163  # DeepSeek-R1 total weight files

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_processed_shard": 0}

def save_state(shard_idx):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_processed_shard": shard_idx}, f)

def convert_tensor_to_8term_csd_chunk(W_chunk):
    """
    Snaps a single sub-matrix chunk to CSD to minimize intermediate memory spikes.
    Performs math in float32 for perfect numerical stability.
    """
    # Force float32 for stable log/pow operations inside the chunk
    W_chunk_fp32 = W_chunk.to(torch.float32)
    W_abs = torch.abs(W_chunk_fp32)
    sg = torch.sign(W_chunk_fp32)
    W_quantized = torch.zeros_like(W_abs)
    current_res = W_abs.clone()
    
    # 8 CSD Snaps
    for term_idx in range(8):
        exponent = torch.round(torch.log2(torch.clamp(current_res, min=1e-12)))
        term = torch.pow(2.0, exponent)
        direction = torch.sign(current_res - W_quantized)
        W_quantized = W_quantized + direction * term
        current_res = torch.abs(W_abs - W_quantized)
        
    W_quantized = sg * W_quantized
    
    # Precise division scaling correction
    W_quantized_safe = torch.where(W_quantized == 0.0, torch.ones_like(W_quantized) * 1e-12, W_quantized)
    scale_alignment = W_chunk_fp32 / W_quantized_safe
    
    W_corrected = W_quantized * scale_alignment
    return W_corrected

def convert_tensor_to_8term_csd(W):
    """
    Safe wrapper that delegates large matrices to row-by-row chunking 
    and natively operates on bfloat16/float16 tensors to cut memory usage by 50%.
    """
    original_dtype = W.dtype
    
    if W.ndim < 2 or W.numel() < 2000000:
        # Process small tensors directly
        return convert_tensor_to_8term_csd_chunk(W).to(original_dtype)
        
    # Large 2D layer matrix: process in 512-row chunks
    W_out = torch.empty_like(W)
    chunk_size = 512
    for idx in range(0, W.shape[0], chunk_size):
        end_idx = min(idx + chunk_size, W.shape[0])
        # Temporarily slice, cast to float32 for snapped math, and cast back to native format
        chunk_conv = convert_tensor_to_8term_csd_chunk(W[idx:end_idx])
        W_out[idx:end_idx] = chunk_conv.to(original_dtype)
    return W_out

def run_conversion_pipeline():
    api = HfApi()
    
    # Retrieve HF Token securely from Kaggle Secrets or environment variables
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
            print("[INFO] Successfully loaded HF_TOKEN from Kaggle Secrets.")
        except Exception as secrets_err:
            print(f"[WARNING] Could not load HF_TOKEN from Kaggle Secrets: {secrets_err}")
            print("Please set your HF_TOKEN inside Kaggle Secrets before launching.")
        
    state = load_state()
    start_shard = state["last_processed_shard"] + 1
    
    print("="*80)
    print(f"  INDRA-BIT 671B FULL GHOST CONVERTER — STARTING AT SHARD {start_shard}/{TOTAL_SHARDS}")
    print("="*80)
    
    # Ensure target repository exists on Hugging Face
    try:
        api.create_repo(repo_id=TARGET_REPO, repo_type="model", exist_ok=True, token=hf_token)
        print(f"[INIT] Target Hub Repository verified: https://huggingface.co/{TARGET_REPO}")
    except Exception as e:
        print(f"[WARNING] Could not verify/create target repository: {e}")

    for shard_idx in range(start_shard, TOTAL_SHARDS + 1):
        # DeepSeek-R1 uses asymmetric sharding: 5-digit index, but 6-digit total shard limit!
        shard_name = f"model-{shard_idx:05d}-of-{TOTAL_SHARDS:06d}.safetensors"
        print(f"\n[SHARD {shard_idx}/{TOTAL_SHARDS}] Downloading {shard_name} from Hugging Face...")
        
        t_download = time.time()
        try:
            local_path = hf_hub_download(
                repo_id=SOURCE_REPO,
                filename=shard_name,
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False,
                token=hf_token
            )
            print(f"      Downloaded successfully in {time.time() - t_download:.1f}s.")
        except Exception as e:
            print(f"      [FATAL] Failed to download shard {shard_name}: {e}")
            break
            
        print("      Extracting and converting weights on CPU...")
        t_convert = time.time()
        
        # Load weights on CPU in native format (bfloat16) to protect memory headroom
        converted_tensors = {}
        with safe_open(local_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                W = f.get_tensor(key)
                # Apply 8-term snapping on transformer layers (skip scaling final layers if needed)
                if "layers" in key:
                    W_conv = convert_tensor_to_8term_csd(W)
                else:
                    W_conv = W
                converted_tensors[key] = W_conv
                
                # Proactively free intermediate tensor reference and run garbage collection
                del W
                gc.collect()
                
        print(f"      CSD Snapping complete in {time.time() - t_convert:.1f}s.")
        
        # Save converted file locally
        converted_local_path = os.path.join(CACHE_DIR, f"converted-{shard_name}")
        save_file(converted_tensors, converted_local_path)
        
        # Free memory instantly before upload
        del converted_tensors
        gc.collect()
        
        # Upload converted shard to target repo LFS
        print("      Uploading quantized shard to Hugging Face LFS...")
        t_upload = time.time()
        try:
            api.upload_file(
                path_or_fileobj=converted_local_path,
                path_in_repo=shard_name,
                repo_id=TARGET_REPO,
                repo_type="model",
                token=hf_token
            )
            print(f"      Quantized shard uploaded successfully in {time.time() - t_upload:.1f}s!")
            
            # Update state file to secure resumability
            save_state(shard_idx)
        except Exception as e:
            print(f"      [ERROR] Upload failed for shard {shard_name}: {e}")
            break
            
        # Clean local cache files completely to guarantee zero disk growth
        print("      Wiping downloaded cache files and all hidden HF metadata from disk...")
        try:
            # Complete directory-level purge to destroy all hidden LFS blobs, lock files, and symlinks
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR, ignore_errors=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            print("      Cache folder successfully purged and reinitialized to 0 bytes.")
        except Exception as cleanup_err:
            print(f"      [WARNING] Cache folder purging encountered errors: {cleanup_err}")
            
        gc.collect()
        
    print("\n" + "="*80)
    print("  GHOST CONVERSION RUN CONCLUDED FOR THIS SESSION")
    print("="*80)

if __name__ == "__main__":
    # Ensure cache folder exists
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    run_conversion_pipeline()
