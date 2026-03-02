import os
import ast
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

# ── paths ──────────────────────────────────────────────────────────────────
# everything gets saved relative to wherever this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
IMG_EMB_PATH = os.path.join(SCRIPT_DIR, "img_embeddings.npy")
TXT_EMB_PATH = os.path.join(SCRIPT_DIR, "txt_embeddings.npy")
META_PATH    = os.path.join(SCRIPT_DIR, "metadata.json")

# the flickr 1k test set hosted on huggingface
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
    # each row is one image; columns include 'filename' and 'raw' (a JSON list of 5 captions)

    image_filenames = df["filename"].tolist()   # list of 1000 strings like "1007129816.jpg"

    all_captions = []
    for _, row in df.iterrows():
        # 'raw' is stored as a JSON-encoded list of 5 caption strings
        caps = ast.literal_eval(row["raw"])
        # extend so that captions for image i are at indices 5*i .. 5*i+4
        all_captions.extend(caps)

    print(f"loaded {len(image_filenames)} images and {len(all_captions)} captions")
    assert len(image_filenames) == 1000, "expected exactly 1000 images"
    assert len(all_captions) == 5000,   "expected 5 captions per image = 5000 total"

    return image_filenames, all_captions


def extract_images(zip_path, image_filenames):
    """Unzip the images into ./images/ — skips files that are already there."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("extracting images from zip...")
    with zipfile.ZipFile(zip_path) as zf:
        for fname in image_filenames:
            dest = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(dest):
                continue   # already extracted, don't redo it
            # inside the zip they live under this prefix
            zip_inner_path = "images_flickr_1k_test/" + fname
            with zf.open(zip_inner_path) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    print(f"images saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    csv_path, zip_path = load_dataset_files()
    image_filenames, all_captions = parse_csv(csv_path)
    extract_images(zip_path, image_filenames)

    # just exploring what the data looks like for now
    print("\nfirst 3 filenames:", image_filenames[:3])
    print("captions for image 0:")
    for cap in all_captions[:5]:
        print(" ", cap)
