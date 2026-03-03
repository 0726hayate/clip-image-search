import os
import ast
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR   = os.path.join(SCRIPT_DIR, "images")
IMG_EMB_PATH = os.path.join(SCRIPT_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(SCRIPT_DIR, "txt_embeddings.npy")
META_PATH    = os.path.join(SCRIPT_DIR, "metadata.json")

# ── model / batch settings ─────────────────────────────────────────────────
MODEL_NAME  = "clip-ViT-B-32"  # 512-dim embeddings, available in sentence-transformers
IMG_BATCH   = 64               # how many images to encode at once — fits in GPU memory
TXT_BATCH   = 256              # captions are smaller so we can do bigger batches

REPO_ID = "nlphuji/flickr_1k_test_image_text_retrieval"


def load_dataset_files():
    """Download (or fetch from cache) the CSV and image zip."""
    print("fetching dataset CSV from huggingface hub...")
    # hf_hub_download checks the local cache first, so this is instant if already downloaded
    csv_path = hf_hub_download(repo_id=REPO_ID, filename="test_1k_flickr.csv", repo_type="dataset")

    print("fetching image zip from huggingface hub...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename="images_flickr_1k_test.zip", repo_type="dataset")

    return csv_path, zip_path


def parse_csv(csv_path):
    """Read the CSV and pull out filenames and captions."""
    df = pd.read_csv(csv_path)

    image_filenames = df["filename"].tolist()   # 1000 strings like "1007129816.jpg"

    all_captions = []
    for _, row in df.iterrows():
        # 'raw' is a JSON-encoded list of 5 caption strings per image
        caps = ast.literal_eval(row["raw"])
        # after this loop: captions for image i are at all_captions[5*i : 5*i+5]
        all_captions.extend(caps)

    print(f"loaded {len(image_filenames)} images and {len(all_captions)} captions")
    assert len(image_filenames) == 1000
    assert len(all_captions) == 5000

    return image_filenames, all_captions


def extract_images(zip_path, image_filenames):
    """Unzip the images into ./images/ — skips files that are already there."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("extracting images from zip...")
    with zipfile.ZipFile(zip_path) as zf:
        for fname in image_filenames:
            dest = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(dest):
                continue  # skip if we already extracted this one
            zip_inner_path = "images_flickr_1k_test/" + fname
            with zf.open(zip_inner_path) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    print(f"images saved to {IMAGES_DIR}/")


def encode_images(model, image_filenames):
    """Load all images as PIL objects and encode them with CLIP."""
    print("loading images into memory...")
    pil_images = []
    for fname in image_filenames:
        img_path = os.path.join(IMAGES_DIR, fname)
        # open as RGB — some flickr images are grayscale, CLIP expects 3 channels
        img = Image.open(img_path).convert("RGB")
        pil_images.append(img)

    print(f"encoding {len(pil_images)} images (batch_size={IMG_BATCH})...")
    # sentence-transformers handles PIL images natively, no manual preprocessing needed
    img_embeddings = model.encode(
        pil_images,
        batch_size=IMG_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors so cosine sim = dot product later
    )
    # shape: (1000, 512)
    print(f"image embeddings shape: {img_embeddings.shape}")
    return img_embeddings


if __name__ == "__main__":
    csv_path, zip_path = load_dataset_files()
    image_filenames, all_captions = parse_csv(csv_path)
    extract_images(zip_path, image_filenames)

    print(f"\nloading CLIP model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    img_embeddings = encode_images(model, image_filenames)

    # save image embeddings — text embeddings coming next
    np.save(IMG_EMB_PATH, img_embeddings)
    print(f"saved image embeddings to {IMG_EMB_PATH}")
    print("done with images. will add text encoding next.")
