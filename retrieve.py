import os
import json

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_EMB_PATH = os.path.join(SCRIPT_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(SCRIPT_DIR, "txt_embeddings.npy")
META_PATH    = os.path.join(SCRIPT_DIR, "metadata.json")


def load_embeddings():
    """Load embeddings and metadata saved by embed.py."""
    print("loading embeddings...")
    img_embs = np.load(IMG_EMB_PATH)   # (1000, 512), L2-normalized
    txt_embs = np.load(TXT_EMB_PATH)   # (5000, 512), L2-normalized

    with open(META_PATH) as f:
        meta = json.load(f)

    filenames = meta["filenames"]   # list of 1000 jpg filenames
    captions  = meta["captions"]    # list of 5000 caption strings

    print(f"image embeddings: {img_embs.shape}")
    print(f"text embeddings:  {txt_embs.shape}")
    return img_embs, txt_embs, filenames, captions


def recall_at_k_text_to_image(txt_embs, img_embs, ks=(1, 5, 10)):
    """Recall@K for text-to-image retrieval.

    For each of the 5000 captions, compute similarity to all 1000 images
    and check if the ground truth image lands in the top K.

    Ground truth: caption at flat index t belongs to image t // 5.
    """
    # compute full similarity matrix all at once — shape (5000, 1000)
    # dot product = cosine sim because embeddings are already L2-normalized
    sim = txt_embs @ img_embs.T

    hits = {k: 0 for k in ks}

    for t in range(len(txt_embs)):
        # the correct image for caption t is always at index t // 5
        # (captions 0-4 go with image 0, captions 5-9 go with image 1, etc.)
        gt_img = t // 5

        # rank all images by similarity to this caption, descending
        ranked = np.argsort(-sim[t])   # shape (1000,)

        for k in ks:
            # hit if the ground truth image is in the top k
            if gt_img in ranked[:k]:
                hits[k] += 1

    # convert hit counts to percentages
    recall = {k: hits[k] / len(txt_embs) * 100 for k in ks}
    return recall


def recall_at_k_image_to_text(img_embs, txt_embs, ks=(1, 5, 10)):
    """Recall@K for image-to-text retrieval.

    For each of the 1000 images, check if any of its 5 ground truth captions
    land in the top K when ranking all 5000 captions by similarity.
    """
    # similarity matrix — shape (1000, 5000)
    sim = img_embs @ txt_embs.T

    hits = {k: 0 for k in ks}

    for i in range(len(img_embs)):
        # image i has 5 paired captions at these flat indices
        gt_captions = set(range(5 * i, 5 * i + 5))

        # rank all 5000 captions by similarity to this image
        ranked = np.argsort(-sim[i])   # shape (5000,)

        for k in ks:
            top_k_set = set(ranked[:k].tolist())
            # success if at least one ground truth caption appears in the top k
            if len(gt_captions & top_k_set) > 0:
                hits[k] += 1

    recall = {k: hits[k] / len(img_embs) * 100 for k in ks}
    return recall


if __name__ == "__main__":
    img_embs, txt_embs, filenames, captions = load_embeddings()

    ks = (1, 5, 10)

    print("\nrunning text -> image evaluation (5000 queries)...")
    ti = recall_at_k_text_to_image(txt_embs, img_embs, ks=ks)

    print("running image -> text evaluation (1000 queries)...")
    it = recall_at_k_image_to_text(img_embs, txt_embs, ks=ks)

    # print a clean table
    print("\n" + "=" * 48)
    print(f"  {'task':<20} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print("-" * 48)
    print(f"  {'text -> image':<20} {ti[1]:>6.1f}% {ti[5]:>6.1f}% {ti[10]:>6.1f}%")
    print(f"  {'image -> text':<20} {it[1]:>6.1f}% {it[5]:>6.1f}% {it[10]:>6.1f}%")
    print("=" * 48)
    print("\nzero-shot CLIP ViT-B/32 on Flickr30K 1K test set.")
