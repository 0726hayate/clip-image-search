import os
import ast
import json
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor, AutoModel

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR   = os.path.join(SCRIPT_DIR, "images")
META_PATH    = os.path.join(SCRIPT_DIR, "metadata.json")

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "nlphuji/flickr_1k_test_image_text_retrieval"

# ── model registry ─────────────────────────────────────────────────────────
MODELS = {
    "b32":  "openai/clip-vit-base-patch32",
    "l14":  "openai/clip-vit-large-patch14",
    "jina": "jinaai/jina-clip-v2",
    "h14":  "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
}
JINA_MODELS = {"jina"}

# smaller batches for bigger models to stay within GPU memory
IMG_BATCH = {"b32": 64, "l14": 32, "jina": 32, "h14": 16}
TXT_BATCH = {"b32": 256, "l14": 128, "jina": 128, "h14": 64}


def load_dataset_files():
    """Download (or fetch from cache) the CSV and image zip."""
    print("fetching dataset CSV from huggingface hub...")
    csv_path = hf_hub_download(repo_id=REPO_ID, filename="test_1k_flickr.csv", repo_type="dataset")

    print("fetching image zip from huggingface hub...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename="images_flickr_1k_test.zip", repo_type="dataset")

    return csv_path, zip_path


def parse_csv(csv_path):
    """Read the CSV and return parallel lists of filenames and captions."""
    df = pd.read_csv(csv_path)
    image_filenames = df["filename"].tolist()

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
                continue
            zip_inner_path = "images_flickr_1k_test/" + fname
            with zf.open(zip_inner_path) as src, open(dest, "wb") as dst:
                dst.write(src.read())
    print(f"images saved to {IMAGES_DIR}/")


def encode_clip(model_name, pil_images, captions, img_batch, txt_batch):
    """Encode with HuggingFace CLIPModel + CLIPProcessor."""
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    # projection_dim tells us the output embedding size — never hardcode it
    dim = model.config.projection_dim
    print(f"  loaded | projection_dim={dim} | device={DEVICE}")

    img_embs = []
    with torch.no_grad():
        for i in range(0, len(pil_images), img_batch):
            batch  = pil_images[i : i + img_batch]
            inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
            feats  = model.get_image_features(**inputs)
            img_embs.append(F.normalize(feats, dim=-1).cpu().numpy())
    img_embs = np.vstack(img_embs)
    print(f"  image embeddings: {img_embs.shape}")

    txt_embs = []
    with torch.no_grad():
        for i in range(0, len(captions), txt_batch):
            batch  = captions[i : i + txt_batch]
            inputs = processor(text=batch, return_tensors="pt",
                               padding=True, truncation=True).to(DEVICE)
            feats  = model.get_text_features(**inputs)
            txt_embs.append(F.normalize(feats, dim=-1).cpu().numpy())
    txt_embs = np.vstack(txt_embs)
    print(f"  text embeddings:  {txt_embs.shape}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return img_embs, txt_embs


def encode_jina(model_name, pil_images, captions, img_batch, txt_batch):
    """Encode with Jina CLIP v2 (AutoModel, trust_remote_code=True).

    Jina's encode_image / encode_text return L2-normalized numpy arrays.
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    img_embs = []
    for i in range(0, len(pil_images), img_batch):
        batch = pil_images[i : i + img_batch]
        with torch.no_grad():
            embs = model.encode_image(batch)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().numpy()
        img_embs.append(np.array(embs))
    img_embs = np.vstack(img_embs)
    print(f"  image embeddings: {img_embs.shape}")

    txt_embs = []
    for i in range(0, len(captions), txt_batch):
        batch = captions[i : i + txt_batch]
        with torch.no_grad():
            embs = model.encode_text(batch)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().numpy()
        txt_embs.append(np.array(embs))
    txt_embs = np.vstack(txt_embs)
    print(f"  text embeddings:  {txt_embs.shape}")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return img_embs, txt_embs


if __name__ == "__main__":
    csv_path, zip_path = load_dataset_files()
    image_filenames, all_captions = parse_csv(csv_path)
    extract_images(zip_path, image_filenames)

    # load all images into memory once; reuse across all 4 models
    print("\nloading all images into memory...")
    pil_images = [
        Image.open(os.path.join(IMAGES_DIR, f)).convert("RGB")
        for f in image_filenames
    ]

    # if the original b32 files exist (from old embed.py run), reuse them
    old_img = os.path.join(SCRIPT_DIR, "img_embeddings.npy")
    old_txt = os.path.join(SCRIPT_DIR, "txt_embeddings.npy")
    new_img_b32 = os.path.join(SCRIPT_DIR, "img_embeddings_b32.npy")
    new_txt_b32 = os.path.join(SCRIPT_DIR, "txt_embeddings_b32.npy")
    if os.path.exists(old_img) and not os.path.exists(new_img_b32):
        print("\n[b32] reusing existing img_embeddings.npy -> img_embeddings_b32.npy")
        np.save(new_img_b32, np.load(old_img))
        np.save(new_txt_b32, np.load(old_txt))

    for tag, model_name in MODELS.items():
        img_out = os.path.join(SCRIPT_DIR, f"img_embeddings_{tag}.npy")
        txt_out = os.path.join(SCRIPT_DIR, f"txt_embeddings_{tag}.npy")

        if os.path.exists(img_out) and os.path.exists(txt_out):
            print(f"\n[{tag}] embeddings already exist, skipping")
            continue

        print(f"\n[{tag}] loading {model_name}...")
        encode_fn = encode_jina if tag in JINA_MODELS else encode_clip
        img_embs, txt_embs = encode_fn(
            model_name, pil_images, all_captions,
            IMG_BATCH[tag], TXT_BATCH[tag],
        )

        np.save(img_out, img_embs)
        np.save(txt_out, txt_embs)
        print(f"  saved -> {img_out}")
        print(f"  saved -> {txt_out}")

    # save metadata (same for all models — just filenames and captions)
    with open(META_PATH, "w") as f:
        json.dump({"filenames": image_filenames, "captions": all_captions}, f)

    print("\ndone. run retrieve.py next.")
