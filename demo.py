import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_EMB_PATH = os.path.join(SCRIPT_DIR, "img_embeddings.npy")
META_PATH    = os.path.join(SCRIPT_DIR, "metadata.json")
IMAGES_DIR   = os.path.join(SCRIPT_DIR, "images")

MODEL_NAME = "clip-ViT-B-32"
TOP_K      = 5   # number of results to show per query


def main():
    # load pre-computed image embeddings so we don't need to re-encode on every run
    print("loading image embeddings...")
    img_embs = np.load(IMG_EMB_PATH)   # (1000, 512), L2-normalized

    with open(META_PATH) as f:
        meta = json.load(f)
    filenames = meta["filenames"]   # 1000 jpg filenames
    captions  = meta["captions"]    # 5000 captions (5 per image)

    # load CLIP model — only needed to encode the user's text query
    print(f"loading CLIP model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    print("\ntype a text query to find matching images.")
    print("type 'quit' to exit.\n")

    while True:
        query = input("query: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # encode the query into the same 512-dim space as the images
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]   # shape (512,)

        # dot product against all 1000 image embeddings = cosine similarity
        scores = q_emb @ img_embs.T   # shape (1000,)

        # get top-K image indices sorted by score descending
        top_k = np.argsort(-scores)[:TOP_K]

        print(f"\ntop {TOP_K} results for: \"{query}\"")
        print("-" * 55)
        for rank, img_idx in enumerate(top_k, start=1):
            score = scores[img_idx]
            # show the first of the 5 ground truth captions as a reference description
            sample_caption = captions[5 * img_idx]
            img_path = os.path.join(IMAGES_DIR, filenames[img_idx])
            print(f"  {rank}. {filenames[img_idx]}  (score: {score:.4f})")
            print(f"     \"{sample_caption}\"")
        print()


if __name__ == "__main__":
    main()
