import os
import json

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
META_PATH  = os.path.join(SCRIPT_DIR, "metadata.json")

# ── model registry (same order as embed.py) ────────────────────────────────
MODEL_LABELS = {
    "b32":  ("baseline",           "CLIP ViT-B/32"),
    "l14":  ("exp1: better vision","CLIP ViT-L/14"),
    "jina": ("exp2: better text",  "Jina CLIP v2"),
    "h14":  ("exp3: both scaled",  "OpenCLIP ViT-H/14"),
}


def recall_at_k_text_to_image(txt_embs, img_embs, ks=(1, 5, 10)):
    """Recall@K for text-to-image retrieval.

    Caption t belongs to image t // 5 (1000 images × 5 captions = 5000 queries).
    """
    sim  = txt_embs @ img_embs.T   # (5000, 1000) — dot product = cosine sim (normalized)
    hits = {k: 0 for k in ks}

    for t in range(len(txt_embs)):
        gt_img = t // 5
        ranked = np.argsort(-sim[t])
        for k in ks:
            if gt_img in ranked[:k]:
                hits[k] += 1

    return {k: hits[k] / len(txt_embs) * 100 for k in ks}


def recall_at_k_image_to_text(img_embs, txt_embs, ks=(1, 5, 10)):
    """Recall@K for image-to-text retrieval.

    Image i has 5 paired captions at flat indices 5i..5i+4.
    """
    sim  = img_embs @ txt_embs.T   # (1000, 5000)
    hits = {k: 0 for k in ks}

    for i in range(len(img_embs)):
        gt_captions = set(range(5 * i, 5 * i + 5))
        ranked      = np.argsort(-sim[i])
        for k in ks:
            if len(gt_captions & set(ranked[:k].tolist())) > 0:
                hits[k] += 1

    return {k: hits[k] / len(img_embs) * 100 for k in ks}


if __name__ == "__main__":
    with open(META_PATH) as f:
        meta = json.load(f)

    ks = (1, 5, 10)
    results = {}

    for tag, (experiment, label) in MODEL_LABELS.items():
        img_path = os.path.join(SCRIPT_DIR, f"img_embeddings_{tag}.npy")
        txt_path = os.path.join(SCRIPT_DIR, f"txt_embeddings_{tag}.npy")

        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            print(f"[{tag}] embeddings not found, skipping (run embed.py first)")
            continue

        img_embs = np.load(img_path)   # (1000, D), L2-normalized
        txt_embs = np.load(txt_path)   # (5000, D), L2-normalized

        print(f"[{tag}] {label} | dim={img_embs.shape[1]}")
        ti = recall_at_k_text_to_image(txt_embs, img_embs, ks=ks)
        it = recall_at_k_image_to_text(img_embs, txt_embs, ks=ks)
        results[tag] = (experiment, label, ti, it)

    # ── comparison table ──────────────────────────────────────────────────
    w = 24
    print("\n" + "=" * 72)
    print(f"  {'experiment':<18} {'model':<22} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print("  text → image retrieval (Flickr30K 1K test set, 5000 queries)")
    print("-" * 72)
    for tag, (experiment, label, ti, it) in results.items():
        print(f"  {experiment:<18} {label:<22} {ti[1]:>6.1f}% {ti[5]:>6.1f}% {ti[10]:>6.1f}%")
    print("=" * 72)

    print("\n" + "=" * 72)
    print(f"  {'experiment':<18} {'model':<22} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print("  image → text retrieval (1000 queries)")
    print("-" * 72)
    for tag, (experiment, label, ti, it) in results.items():
        print(f"  {experiment:<18} {label:<22} {it[1]:>6.1f}% {it[5]:>6.1f}% {it[10]:>6.1f}%")
    print("=" * 72)
