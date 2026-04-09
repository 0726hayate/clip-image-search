"""Four benchmark tables for the Adapter Swarm pattern.

Run after demo_swarm.py --build (which produces swarm_index_*.npy files)
and after the niche LoRAs are trained (or 5-fold CV completes).

Tables:
 1. Storage           — base + N adapters vs N full FT models
 2. Latency           — cold base load, hot adapter swap, query encode, search
 3. Quality (img→img) — base vs FT R@K per niche, aggregated across folds if present
 4. Quality (txt→img) — prompt-ensembled P@K per niche, mirrors the demo path
"""

import os
import glob
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

from finetune import recall_at_k_series, build_class_idx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BASE_NAME  = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
NICHES     = ["gundam", "pokemon", "paintings"]
HF_CACHE   = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

# ── Prompt templates per niche (3 each, embeddings averaged for stability) ──
# {C} is replaced with the display name from CLASS_DISPLAY below.
PROMPT_TEMPLATES = {
    "gundam":    ["a Gundam {C} mobile suit",
                  "mobile suit from Mobile Suit Gundam {C}",
                  "mecha from {C}"],
    "pokemon":   ["a {C}-type pokemon",
                  "{C} type pokemon",
                  "official artwork of a {C} pokemon"],
    "paintings": ["a {C} painting",
                  "art in the {C} style",
                  "a painting in the {C} movement"],
}

# Pretty display names for the underscored / abbreviated class strings in labels.json
CLASS_DISPLAY = {
    "gundam":    {"00": "Gundam 00", "age": "AGE", "grg": "Reconguista in G",
                  "ibo": "Iron-Blooded Orphans", "uc": "Universal Century Unicorn",
                  "wfm": "Witch from Mercury"},
    "pokemon":   {},   # lowercase types are fine as-is
    "paintings": {},   # underscores stripped at format time
}


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


def _collect_fold_files(niche):
    """Return list of (base_path, ft_path, labels_path, fold_idx) for any folds present."""
    pattern = os.path.join(SCRIPT_DIR, f"{niche}_val_base_h14_fold*.npy")
    out = []
    for base_p in sorted(glob.glob(pattern)):
        fold_idx = int(base_p.split("_fold")[-1].rsplit(".", 1)[0])
        ft_p  = os.path.join(SCRIPT_DIR, f"{niche}_val_ft_h14_fold{fold_idx}.npy")
        lab_p = os.path.join(SCRIPT_DIR, f"{niche}_val_labels_fold{fold_idx}.npy")
        if os.path.exists(ft_p) and os.path.exists(lab_p):
            out.append((base_p, ft_p, lab_p, fold_idx))
    return out


def quality_table():
    """Image→image R@K per niche. Aggregate across folds if present."""
    print("\n== QUALITY: image→image R@K per niche ==\n")
    print(f"  {'niche':<14} {'mode':<10} {'N':>5} {'base R@1':>14} {'ft R@1':>14} "
          f"{'Δ R@1':>10} {'base R@10':>14} {'ft R@10':>14} {'Δ R@10':>10}")
    print(f"  " + "-" * 110)
    for niche in NICHES:
        folds = _collect_fold_files(niche)
        if folds:
            b1, f1, b10, f10, ns = [], [], [], [], []
            for base_p, ft_p, lab_p, _ in folds:
                base_e = np.load(base_p); ft_e = np.load(ft_p); labs = np.load(lab_p)
                b = recall_at_k_series(base_e, labs)
                f = recall_at_k_series(ft_e, labs)
                b1.append(b[1]);  f1.append(f[1])
                b10.append(b[10]); f10.append(f[10])
                ns.append(len(labs))
            mb1, sb1 = np.mean(b1), np.std(b1)
            mf1, sf1 = np.mean(f1), np.std(f1)
            mb10, sb10 = np.mean(b10), np.std(b10)
            mf10, sf10 = np.mean(f10), np.std(f10)
            n_str = f"{int(round(np.mean(ns)))}"
            print(f"  {niche:<14} {f'{len(folds)}-fold':<10} {n_str:>5} "
                  f"{mb1:>7.1f}±{sb1:>3.1f}%  {mf1:>7.1f}±{sf1:>3.1f}% "
                  f"{mf1-mb1:>+9.1f}pp "
                  f"{mb10:>7.1f}±{sb10:>3.1f}%  {mf10:>7.1f}±{sf10:>3.1f}% "
                  f"{mf10-mb10:>+9.1f}pp")
        else:
            base_p = os.path.join(SCRIPT_DIR, f"{niche}_val_base_h14.npy")
            ft_p   = os.path.join(SCRIPT_DIR, f"{niche}_val_ft_h14.npy")
            lab_p  = os.path.join(SCRIPT_DIR, f"{niche}_val_labels.npy")
            if not all(os.path.exists(p) for p in [base_p, ft_p, lab_p]):
                print(f"  {niche:<14}  (val embeddings not found — train adapter first)")
                continue
            base_e = np.load(base_p); ft_e = np.load(ft_p); labs = np.load(lab_p)
            b = recall_at_k_series(base_e, labs)
            f = recall_at_k_series(ft_e, labs)
            print(f"  {niche:<14} {'1-split':<10} {len(labs):>5d} "
                  f"{b[1]:>13.1f}% {f[1]:>13.1f}% {f[1]-b[1]:>+9.1f}pp "
                  f"{b[10]:>13.1f}% {f[10]:>13.1f}% {f[10]-b[10]:>+9.1f}pp")


def _ensembled_text_embs(model, processor, niche, classes_sorted):
    """For each class in `classes_sorted`, encode all 3 prompt templates and
    average → (n_classes, dim). Class display names come from CLASS_DISPLAY,
    or class string itself with underscores stripped."""
    templates = PROMPT_TEMPLATES[niche]
    display_map = CLASS_DISPLAY.get(niche, {})
    ensembled = []
    for cls in classes_sorted:
        display = display_map.get(cls, cls.replace("_", " "))
        prompts = [t.format(C=display) for t in templates]
        with torch.no_grad():
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
            embs = model.get_text_features(**inputs)
            embs = F.normalize(embs, dim=-1)
            mean_emb = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
        ensembled.append(mean_emb.cpu().numpy()[0])
    return np.array(ensembled)


def _txt_img_pak(text_embs, img_embs, img_labels, ks=(1, 5, 10)):
    """For each text prompt (= class), retrieve top-K images; return mean P@K."""
    sims = text_embs @ img_embs.T   # (n_classes, n_images)
    p_at = {k: [] for k in ks}
    for cls_idx in range(len(text_embs)):
        ranked = np.argsort(-sims[cls_idx])
        for k in ks:
            top_k = ranked[:k]
            hits = int(np.sum(img_labels[top_k] == cls_idx))
            p_at[k].append(hits / k)
    return {k: float(np.mean(p_at[k])) * 100 for k in ks}


def text_quality_table():
    """Text→image P@K per niche. Mirrors the demo's actual flow."""
    print("\n== QUALITY: text→image P@K per niche (3-prompt ensemble per class) ==\n")
    print(f"  {'niche':<14} {'mode':<10} "
          f"{'base P@1':>14} {'ft P@1':>14} {'Δ P@1':>10} "
          f"{'base P@5':>14} {'ft P@5':>14} {'Δ P@5':>10}")
    print(f"  " + "-" * 110)

    print(f"  loading base text encoder ({BASE_NAME})...", end=" ", flush=True)
    processor = CLIPProcessor.from_pretrained(BASE_NAME)
    base = CLIPModel.from_pretrained(BASE_NAME).to(DEVICE)
    base.eval()
    print("done")

    for niche in NICHES:
        labels_json = os.path.join(SCRIPT_DIR, niche, "labels.json")
        if not os.path.exists(labels_json):
            print(f"  {niche:<14}  (labels.json not found, skipping)")
            continue
        with open(labels_json) as f: labels_map = json.load(f)
        classes_sorted = sorted(set(labels_map.values()))
        text_embs = _ensembled_text_embs(base, processor, niche, classes_sorted)

        folds = _collect_fold_files(niche)
        if folds:
            bp1, fp1, bp5, fp5 = [], [], [], []
            for base_p, ft_p, lab_p, _ in folds:
                base_img = np.load(base_p); ft_img = np.load(ft_p); labs = np.load(lab_p)
                b = _txt_img_pak(text_embs, base_img, labs)
                f = _txt_img_pak(text_embs, ft_img,   labs)
                bp1.append(b[1]); fp1.append(f[1])
                bp5.append(b[5]); fp5.append(f[5])
            mb1, sb1 = np.mean(bp1), np.std(bp1)
            mf1, sf1 = np.mean(fp1), np.std(fp1)
            mb5, sb5 = np.mean(bp5), np.std(bp5)
            mf5, sf5 = np.mean(fp5), np.std(fp5)
            print(f"  {niche:<14} {f'{len(folds)}-fold':<10} "
                  f"{mb1:>7.1f}±{sb1:>3.1f}%  {mf1:>7.1f}±{sf1:>3.1f}% "
                  f"{mf1-mb1:>+9.1f}pp "
                  f"{mb5:>7.1f}±{sb5:>3.1f}%  {mf5:>7.1f}±{sf5:>3.1f}% "
                  f"{mf5-mb5:>+9.1f}pp")
        else:
            base_p = os.path.join(SCRIPT_DIR, f"{niche}_val_base_h14.npy")
            ft_p   = os.path.join(SCRIPT_DIR, f"{niche}_val_ft_h14.npy")
            lab_p  = os.path.join(SCRIPT_DIR, f"{niche}_val_labels.npy")
            if not all(os.path.exists(p) for p in [base_p, ft_p, lab_p]):
                print(f"  {niche:<14}  (val embeddings not found)")
                continue
            base_img = np.load(base_p); ft_img = np.load(ft_p); labs = np.load(lab_p)
            b = _txt_img_pak(text_embs, base_img, labs)
            f = _txt_img_pak(text_embs, ft_img,   labs)
            print(f"  {niche:<14} {'1-split':<10} "
                  f"{b[1]:>13.1f}% {f[1]:>13.1f}% {f[1]-b[1]:>+9.1f}pp "
                  f"{b[5]:>13.1f}% {f[5]:>13.1f}% {f[5]-b[5]:>+9.1f}pp")


if __name__ == "__main__":
    storage_table()
    latency_table()
    quality_table()
    text_quality_table()
    print()
