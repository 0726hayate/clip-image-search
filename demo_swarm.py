"""Adapter Swarm demo: one CLIP base + N LoRA adapters specialize per niche.

Pinterest's actual problem: hundreds of niche communities (fashion, food,
anime, art, ...). Fine-tuning a separate CLIP per niche means N copies of
a 5 GB model in your serving fleet. LoRA flips this:
  * one base model in memory
  * a 10 MB adapter per niche
  * swap the adapter at request time

This script demonstrates the pattern with 3 niche adapters on OpenCLIP H/14:
  * gundam_lora_h14    — 6 mobile suit series
  * pokemon_lora_h14   — 8 Pokémon types
  * paintings_lora_h14 — 8 art movements

A subtle but important detail: LoRA only touches the VISION encoder. The
text encoder is frozen across all adapters. So the text embedding for a
query like "fire dragon" is computed ONCE with the base model, then
matched against vision embeddings produced by each niche's adapter — the
shared CLIP text↔image alignment is preserved by addition (LoRA), not
replacement.

Usage:
    # build per-niche image indexes (encode all niche images with their adapter)
    python demo_swarm.py --build

    # search a niche by text
    python demo_swarm.py --niche pokemon   --query "fiery red dragon"
    python demo_swarm.py --niche paintings --query "moonlit sky"
    python demo_swarm.py --niche gundam    --query "white red mobile suit"
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

BASE_NAME  = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
NICHES     = ["gundam", "pokemon", "paintings"]


def load_niche_filelist(niche):
    """Use the existing labels.json from collection time as the niche image set.
    Returns (list of relative paths, dict: rel_path -> class_name)."""
    with open(os.path.join(SCRIPT_DIR, niche, "labels.json")) as f:
        labels_map = json.load(f)
    return list(labels_map.keys()), labels_map


def encode_images(model, processor, rel_paths, batch_size=32):
    """Encode images using a CLIP-style model (works for base or PeftModel)."""
    model.eval()
    all_embs = []
    for i in range(0, len(rel_paths), batch_size):
        batch = rel_paths[i : i + batch_size]
        pils = [Image.open(os.path.join(SCRIPT_DIR, p)).convert("RGB") for p in batch]
        inputs = processor(images=pils, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=pixel_values)
            embs = model.visual_projection(vision_out.pooler_output)
            embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


def encode_text(base_model, processor, query):
    """Text encoder is frozen — encoded once with the base model regardless of niche."""
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = base_model.get_text_features(**inputs)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0]


def build_indexes():
    """Encode every niche's images with its adapter; save embeddings + path lists."""
    print(f"loading base: {BASE_NAME}")
    processor = CLIPProcessor.from_pretrained(BASE_NAME)
    base = CLIPModel.from_pretrained(BASE_NAME).to(DEVICE)

    for niche in NICHES:
        adapter_dir = os.path.join(SCRIPT_DIR, f"{niche}_lora_h14")
        if not os.path.isdir(adapter_dir):
            print(f"[{niche}] adapter not found at {adapter_dir}, skipping")
            continue

        rel_paths, _ = load_niche_filelist(niche)
        print(f"\n[{niche}] {len(rel_paths)} images; loading adapter...")
        ft = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE)
        embs = encode_images(ft, processor, rel_paths)

        np.save(os.path.join(SCRIPT_DIR, f"swarm_index_{niche}_embs.npy"), embs)
        with open(os.path.join(SCRIPT_DIR, f"swarm_index_{niche}_paths.json"), "w") as f:
            json.dump(rel_paths, f)
        print(f"[{niche}] saved index ({embs.shape})")

        # detach adapter so the next iteration starts from clean base
        ft.unload()
        del ft

    print("\nall indexes built. now run with --niche <name> --query <text>")


def search(niche, query, top_k=5):
    """Encode the query with base text encoder, retrieve top-K from niche index."""
    embs_path = os.path.join(SCRIPT_DIR, f"swarm_index_{niche}_embs.npy")
    paths_path = os.path.join(SCRIPT_DIR, f"swarm_index_{niche}_paths.json")
    if not (os.path.exists(embs_path) and os.path.exists(paths_path)):
        print(f"index for niche '{niche}' not built. run with --build first.")
        sys.exit(1)

    print(f"loading base text encoder ({BASE_NAME})...")
    processor = CLIPProcessor.from_pretrained(BASE_NAME)
    t0 = time.time()
    base = CLIPModel.from_pretrained(BASE_NAME).to(DEVICE)
    print(f"  base loaded in {time.time()-t0:.2f}s")

    embs = np.load(embs_path)
    with open(paths_path) as f:
        paths = json.load(f)

    t0 = time.time()
    qvec = encode_text(base, processor, query)
    enc_ms = (time.time() - t0) * 1000

    sims = embs @ qvec
    top_idx = np.argsort(-sims)[:top_k]
    print(f"\nquery: \"{query}\"  niche: {niche}")
    print(f"text encode: {enc_ms:.1f} ms  |  top-{top_k} of {len(paths)} images:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank}. sim={sims[i]:.3f}  {paths[i]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build",  action="store_true",
                   help="encode all niche images with their adapters and save indexes")
    p.add_argument("--niche",  choices=NICHES, help="niche to search")
    p.add_argument("--query",  type=str, help="text query")
    p.add_argument("--top-k",  type=int, default=5)
    args = p.parse_args()

    if args.build:
        build_indexes()
    elif args.niche and args.query:
        search(args.niche, args.query, args.top_k)
    else:
        p.print_help()
