import os
import json

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_EMB_PATH = os.path.join(SCRIPT_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(SCRIPT_DIR, "txt_embeddings.npy")
META_PATH    = os.path.join(SCRIPT_DIR, "metadata.json")

TOP_K = 5  # how many results to show per query


def load_embeddings():
    """Load embeddings and metadata saved by embed.py."""
    print("loading embeddings...")
    img_embs = np.load(IMG_EMB_PATH)   # (1000, 512) — L2 normalized
    txt_embs = np.load(TXT_EMB_PATH)   # (5000, 512) — L2 normalized

    with open(META_PATH) as f:
        meta = json.load(f)

    filenames = meta["filenames"]   # list of 1000 jpg filenames
    captions  = meta["captions"]    # list of 5000 caption strings

    print(f"image embeddings: {img_embs.shape}")
    print(f"text embeddings:  {txt_embs.shape}")
    return img_embs, txt_embs, filenames, captions


def text_to_image(query_embs, img_embs, top_k=TOP_K):
    """Given text embeddings, return top-k image indices for each query.

    Since embeddings are already L2-normalized, dot product = cosine similarity.
    """
    # multiply each text vector against all 1000 image vectors at once
    # result shape: (num_queries, 1000) — each row is similarity scores for one query
    sim = query_embs @ img_embs.T

    # argsort descending: ranked[i] = image indices sorted by similarity for query i
    ranked = np.argsort(-sim, axis=1)

    # return only the top-k columns
    return ranked[:, :top_k]


if __name__ == "__main__":
    img_embs, txt_embs, filenames, captions = load_embeddings()

    # try a few example queries to see if retrieval makes sense
    # pick 3 random captions and see if CLIP retrieves the right image
    test_indices = [0, 100, 500]   # caption indices to query with

    print("\nrunning a few example queries...\n")
    for t in test_indices:
        gt_image = t // 5   # caption t belongs to image t//5

        # get top-k images for this single caption
        q_emb = txt_embs[t:t+1]   # shape (1, 512) — keep 2D for the matmul
        top_k_images = text_to_image(q_emb, img_embs, top_k=TOP_K)[0]  # shape (top_k,)

        print(f"query caption: \"{captions[t]}\"")
        print(f"ground truth image: {filenames[gt_image]} (index {gt_image})")
        print(f"top {TOP_K} retrieved images:")
        for rank, img_idx in enumerate(top_k_images, start=1):
            hit = " <-- correct" if img_idx == gt_image else ""
            print(f"  {rank}. {filenames[img_idx]} (index {img_idx}){hit}")
        print()
