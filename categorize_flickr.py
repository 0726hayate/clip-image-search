import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMG_EMB_PATH = os.path.join(SCRIPT_DIR, "img_embeddings_b32.npy")
OUT_PATH     = os.path.join(SCRIPT_DIR, "flickr_categories.npy")

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── category definitions ──────────────────────────────────────────────────────
# "a photo of ..." framing matches CLIP's training distribution
CATEGORIES = [
    ("people",       "a photo of people or a social scene with people"),
    ("animals",      "a photo of an animal or multiple animals"),
    ("sports",       "a photo of a sporting event or outdoor recreational activity"),
    ("nature",       "a photo of a natural landscape or outdoor scenery without people"),
    ("food",         "a photo of food, drink, or a meal"),
    ("vehicles",     "a photo of a vehicle, car, bicycle, or transportation"),
    ("architecture", "a photo of a building, structure, or urban architecture"),
]

CATEGORY_NAMES  = [name  for name, _      in CATEGORIES]
CATEGORY_PROMPTS = [prompt for _, prompt in CATEGORIES]


if __name__ == "__main__":
    print(f"loading image embeddings from {IMG_EMB_PATH}...")
    img_embs = np.load(IMG_EMB_PATH)   # (1000, 512), already L2-normalized
    print(f"  {img_embs.shape[0]} images, dim={img_embs.shape[1]}")

    print(f"\nloading CLIP ({MODEL_NAME}) to encode category prompts...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # encode all 7 category prompts into the same embedding space
    with torch.no_grad():
        inputs    = processor(text=CATEGORY_PROMPTS, return_tensors="pt",
                              padding=True).to(DEVICE)
        cat_embs  = model.get_text_features(**inputs)
        cat_embs  = F.normalize(cat_embs, dim=-1).cpu().numpy()   # (7, 512)

    # cosine similarity: img (1000, 512) @ cat.T (512, 7) → (1000, 7)
    sim       = img_embs @ cat_embs.T
    # assign each image to the highest-similarity category
    labels    = np.argmax(sim, axis=1)   # (1000,) with values 0..6

    counts = {name: int((labels == i).sum()) for i, name in enumerate(CATEGORY_NAMES)}
    print("\ncategory distribution:")
    for name, count in counts.items():
        bar = "#" * (count // 5)
        print(f"  {name:<14} {count:4d}  {bar}")

    np.save(OUT_PATH, labels)
    print(f"\nsaved -> {OUT_PATH}")
    print("done. run visualize.py next.")
