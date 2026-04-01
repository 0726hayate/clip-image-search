"""Encode 1000 Imagenette images with each of 4 models, base AND LoRA-loaded.

Imagenette has 10 ImageNet classes with built-in labels — no prompting hack
needed. We use the validation split (smaller, ~3925 images) and sample 100
per class for a balanced 1000-image set, matching the old Flickr count.

For each model: encode the same 1000 images twice — once with the base model,
once with the gundam_lora_{tag} adapter loaded on top. The point is to verify
visually (in the t-SNE) that LoRA fine-tuning on Gundam doesn't destroy
general-purpose semantic clustering.

Outputs:
    imagenette_base_{tag}.npy  (4 files, one per model)
    imagenette_ft_{tag}.npy    (4 files, one per model)
    imagenette_labels.npy      (one file, shared across models)
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor, AutoModel
from peft import PeftModel

from finetune import (
    MODEL_CONFIGS, encode_images_clip, encode_images_jina, DEVICE
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PER_CLASS  = 100   # 10 classes × 100 = 1000 images

# Imagenette class names (idx 0-9), pulled from the dataset's ClassLabel feature
IMAGENETTE_LABELS = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]


def sample_imagenette(per_class=PER_CLASS):
    """Return (list of PIL images, np.array of int labels 0-9)."""
    print(f"loading johnowhitaker/imagenette2-320 (streaming)...")
    ds = load_dataset("johnowhitaker/imagenette2-320", split="train", streaming=True)

    by_class = {i: [] for i in range(10)}
    for ex in ds:
        cls = ex["label"]
        if len(by_class[cls]) < per_class:
            by_class[cls].append(ex["image"])
        if all(len(v) >= per_class for v in by_class.values()):
            break

    pils, labels = [], []
    for cls in range(10):
        pils.extend(by_class[cls])
        labels.extend([cls] * len(by_class[cls]))

    print(f"  collected {len(pils)} images across 10 classes "
          f"({[len(by_class[i]) for i in range(10)]})")
    return pils, np.array(labels)


def encode_with_clip_base_and_ft(model_name, adapter_dir, pils):
    """Load base CLIP, encode images; load adapter, encode again."""
    processor = CLIPProcessor.from_pretrained(model_name)
    base = CLIPModel.from_pretrained(model_name).to(DEVICE)

    print(f"  encoding base...")
    base_embs, _ = encode_images_clip(base, processor, pils)

    print(f"  loading adapter {adapter_dir} and encoding ft...")
    ft = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE)
    ft_embs, _ = encode_images_clip(ft, processor, pils)

    del base, ft
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return base_embs, ft_embs


def encode_with_jina_base_and_ft(model_name, adapter_dir, pils):
    base = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
    base.eval()

    print(f"  encoding base...")
    base_embs, _ = encode_images_jina(base, pils)

    print(f"  loading adapter {adapter_dir} and encoding ft...")
    ft = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE)
    ft.eval()
    ft_embs, _ = encode_images_jina(ft, pils)

    del base, ft
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return base_embs, ft_embs


if __name__ == "__main__":
    pils, labels = sample_imagenette()
    np.save(os.path.join(SCRIPT_DIR, "imagenette_labels.npy"), labels)

    for tag, cfg in MODEL_CONFIGS.items():
        adapter_dir = os.path.join(SCRIPT_DIR, f"gundam_lora_{tag}")
        if not os.path.isdir(adapter_dir):
            print(f"\n[{tag}] adapter {adapter_dir} not found, skipping")
            continue

        print(f"\n[{tag}] {cfg['name']}")
        if cfg["api"] == "clip":
            base_e, ft_e = encode_with_clip_base_and_ft(cfg["name"], adapter_dir, pils)
        else:
            base_e, ft_e = encode_with_jina_base_and_ft(cfg["name"], adapter_dir, pils)

        np.save(os.path.join(SCRIPT_DIR, f"imagenette_base_{tag}.npy"), base_e)
        np.save(os.path.join(SCRIPT_DIR, f"imagenette_ft_{tag}.npy"),   ft_e)
        print(f"  saved imagenette_base_{tag}.npy  ({base_e.shape})")
        print(f"  saved imagenette_ft_{tag}.npy    ({ft_e.shape})")

    print("\ndone. run visualize.py next.")
