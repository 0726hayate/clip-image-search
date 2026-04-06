"""Three benchmark tables for the Adapter Swarm pattern.

Run after demo_swarm.py --build (which produces swarm_index_*.npy files)
and after the niche LoRAs are trained.

Tables:
 1. Storage      — base + N adapters vs N full FT models
 2. Latency      — cold base load, hot adapter swap, query encode, search
 3. Quality      — base vs FT R@K per niche (from val embeddings)
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

from finetune import recall_at_k_series

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE_NAME  = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
NICHES     = ["gundam", "pokemon", "paintings"]
HF_CACHE   = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def fmt_bytes(n):
    if n >= 1 << 30: return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20: return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10: return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return total


def find_base_size():
    """OpenCLIP H/14 weights are cached under huggingface hub by repo name."""
    repo = "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    cache_dir = os.path.join(HF_CACHE, repo)
    if not os.path.isdir(cache_dir):
        return None
    return dir_size(cache_dir)


def storage_table():
    print("\n== STORAGE ==\n")
    base_size = find_base_size()
    if base_size is None:
        print("  (base weights not found in HF cache — run demo_swarm.py --build first)")
        return

    adapter_sizes = {}
    for niche in NICHES:
        path = os.path.join(SCRIPT_DIR, f"{niche}_lora_h14")
        if os.path.isdir(path):
            adapter_sizes[niche] = dir_size(path)

    swarm_total = base_size + sum(adapter_sizes.values())
    full_total  = base_size * len(adapter_sizes)

    print(f"  base h14 (one shared copy):        {fmt_bytes(base_size)}")
    for niche, sz in adapter_sizes.items():
        print(f"  adapter {niche}_lora_h14:               {fmt_bytes(sz)}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  swarm total (1 base + {len(adapter_sizes)} adapters): {fmt_bytes(swarm_total)}")
    print(f"  alternative: {len(adapter_sizes)} full FT models:    {fmt_bytes(full_total)}")
    print(f"  savings:     {(1 - swarm_total/full_total)*100:.1f}% "
          f"({full_total/swarm_total:.1f}× storage efficiency)")


def latency_table():
    print("\n== LATENCY ==\n")
    print(f"  {'operation':<48} {'time':>12}")
    print(f"  " + "-" * 60)

    processor = CLIPProcessor.from_pretrained(BASE_NAME)

    # cold load base from disk (warm cache; first call after process start)
    t0 = time.time()
    base = CLIPModel.from_pretrained(BASE_NAME).to(DEVICE)
    cold_load = time.time() - t0
    print(f"  {'cold-load base h14 from cache → GPU':<48} {cold_load*1000:>10.0f} ms")

    # hot adapter swap (base in memory)
    swap_times = []
    for niche in NICHES:
        adapter_dir = os.path.join(SCRIPT_DIR, f"{niche}_lora_h14")
        if not os.path.isdir(adapter_dir): continue
        t0 = time.time()
        ft = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE)
        swap_times.append(time.time() - t0)
        ft.unload(); del ft
    if swap_times:
        avg_swap = sum(swap_times) / len(swap_times) * 1000
        print(f"  {'hot adapter swap (base in memory, 3-niche avg)':<48} {avg_swap:>10.0f} ms")
        print(f"  {'  (vs reloading 3 GB model from disk)':<48} {cold_load*1000:>10.0f} ms each")

    # text encode latency for one query
    t0 = time.time()
    inputs = processor(text=["fire-type dragon"], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = base.get_text_features(**inputs)
        emb = F.normalize(emb, dim=-1)
    text_enc = (time.time() - t0) * 1000
    print(f"  {'encode 1 text query (base text encoder)':<48} {text_enc:>10.1f} ms")

    # search top-5 over a representative niche index
    embs_path = os.path.join(SCRIPT_DIR, "swarm_index_pokemon_embs.npy")
    if os.path.exists(embs_path):
        idx = np.load(embs_path)
        qvec = emb.cpu().numpy()[0]
        t0 = time.time()
        for _ in range(100):
            sims = idx @ qvec
            np.argpartition(-sims, 5)[:5]
        search_ms = (time.time() - t0) / 100 * 1000
        print(f"  {f'search top-5 over {len(idx)} pokemon images (avg of 100)':<48} {search_ms:>10.2f} ms")


def quality_table():
    print("\n== QUALITY (Recall@K per niche) ==\n")
    print(f"  {'niche':<14} {'base R@1':>10} {'ft R@1':>10} {'Δ R@1':>10} "
          f"{'base R@10':>12} {'ft R@10':>10} {'Δ R@10':>10}")
    print(f"  " + "-" * 80)
    for niche in NICHES:
        base_p = os.path.join(SCRIPT_DIR, f"{niche}_val_base_h14.npy")
        ft_p   = os.path.join(SCRIPT_DIR, f"{niche}_val_ft_h14.npy")
        lab_p  = os.path.join(SCRIPT_DIR, f"{niche}_val_labels.npy")
        if not all(os.path.exists(p) for p in [base_p, ft_p, lab_p]):
            print(f"  {niche:<14}  (val embeddings not found — train adapter first)")
            continue
        base_e = np.load(base_p); ft_e = np.load(ft_p); labs = np.load(lab_p)
        b = recall_at_k_series(base_e, labs)
        f = recall_at_k_series(ft_e, labs)
        print(f"  {niche:<14} {b[1]:>9.1f}% {f[1]:>9.1f}% {f[1]-b[1]:>+9.1f}pp "
              f"{b[10]:>11.1f}% {f[10]:>9.1f}% {f[10]-b[10]:>+9.1f}pp")


if __name__ == "__main__":
    storage_table()
    latency_table()
    quality_table()
    print()
