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
MODEL_NAME = "clip-ViT-B-32"  # 512-dim shared image+text embedding space
IMG_BATCH  = 64               # fits comfortably on GPU
TXT_BATCH  = 256              # text is cheaper, bigger batches are fine

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
    """Read the CSV and return parallel lists of filenames and captions."""
    df = pd.read_csv(csv_path)
    # columns: filename, raw (JSON list of 5 captions), and some metadata we don't need

    image_filenames = df["filename"].tolist()   # 1000 jpg filenames

    all_captions = []
    for _, row in df.iterrows():
        # 'raw' is stored as a JSON string containing a list of 5 caption strings
        caps = ast.literal_eval(row["raw"])
        # extend so image i's captions are always at indices 5*i through 5*i+4
        all_captions.extend(caps)

    print(f"loaded {len(image_filenames)} images and {len(all_captions)} captions")
    assert len(image_filenames) == 1000, "expected 1000 images"
    assert len(all_captions) == 5000,   "expected 5 captions per image"

    return image_filenames, all_captions


def extract_images(zip_path, image_filenames):
    """Unzip images to ./images/, skipping ones already extracted."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("extracting images from zip...")
    with zipfile.ZipFile(zip_path) as zf:
        for fname in image_filenames:
            dest = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(dest):
                continue  # already there, skip
            zip_inner_path = "images_flickr_1k_test/" + fname
            with zf.open(zip_inner_path) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    print(f"images saved to {IMAGES_DIR}/")


def encode_images(model, image_filenames):
    """Load all images as PIL and encode them into 512-dim CLIP vectors."""
    print("loading images into memory...")
    pil_images = []
    for fname in image_filenames:
        img_path = os.path.join(IMAGES_DIR, fname)
        # convert to RGB so grayscale images don't break CLIP's 3-channel input
        img = Image.open(img_path).convert("RGB")
        pil_images.append(img)

    print(f"encoding {len(pil_images)} images (batch_size={IMG_BATCH})...")
    # sentence-transformers handles PIL images natively, no manual preprocessing needed
    img_embeddings = model.encode(
        pil_images,
        batch_size=IMG_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors so cosine sim == dot product later
    )
    print(f"image embeddings shape: {img_embeddings.shape}")  # (1000, 512)
    return img_embeddings


def encode_captions(model, all_captions):
    """Encode all 5000 captions into 512-dim CLIP vectors."""
    print(f"encoding {len(all_captions)} captions (batch_size={TXT_BATCH})...")
    txt_embeddings = model.encode(
        all_captions,
        batch_size=TXT_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # same space as images, normalized for dot-product sim
    )
    print(f"text embeddings shape: {txt_embeddings.shape}")  # (5000, 512)
    return txt_embeddings


def save_everything(img_embeddings, txt_embeddings, image_filenames, all_captions):
    """Persist embeddings as .npy and metadata as JSON."""
    np.save(IMG_EMB_PATH, img_embeddings)
    np.save(TXT_EMB_PATH, txt_embeddings)

    # save filenames and captions so retrieve.py can look up results by index
    metadata = {
        "filenames": image_filenames,  # list of 1000 jpg filenames
        "captions": all_captions,      # list of 5000 captions (5 per image, in order)
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print(f"saved:\n  {IMG_EMB_PATH}\n  {TXT_EMB_PATH}\n  {META_PATH}")


if __name__ == "__main__":
    csv_path, zip_path = load_dataset_files()
    image_filenames, all_captions = parse_csv(csv_path)
    extract_images(zip_path, image_filenames)

    print(f"\nloading CLIP model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    img_embeddings = encode_images(model, image_filenames)
    txt_embeddings = encode_captions(model, all_captions)

    save_everything(img_embeddings, txt_embeddings, image_filenames, all_captions)
    print("\ndone. run retrieve.py next to evaluate.")
