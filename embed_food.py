"""Encode 1000 Food-101 images (10 confusable dishes) with each of 4 models, base + LoRA.

Imagenette's 10 ImageNet classes (church, parachute, garbage truck...) are
visually wildly different — baseline CLIP already nails the t-SNE, so the
"LoRA preserves general semantics" check has nothing to test. Food-101 with
a curated 10-class subset (pasta / dessert / sandwich mini-clusters) gives
a much harder fine-grained preservation signal.

10-class subset, 3 visual mini-clusters:
    pasta:    lasagna, ravioli, spaghetti_bolognese, spaghetti_carbonara
    dessert:  chocolate_cake, chocolate_mousse, tiramisu
    sandwich: hamburger, club_sandwich, pulled_pork_sandwich

Outputs:
    food_base_{tag}.npy  (4 files, one per model)
    food_ft_{tag}.npy    (4 files, one per model)
    food_labels.npy      (one file, shared)
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

# Food-101 official class names (the dataset's ClassLabel uses these strings).
FOOD_CLASSES = [
    "lasagna", "ravioli", "spaghetti_bolognese", "spaghetti_carbonara",  # pasta
    "chocolate_cake", "chocolate_mousse", "tiramisu",                    # dessert
    "hamburger", "club_sandwich", "pulled_pork_sandwich",                # sandwich
]


def sample_food(per_class=PER_CLASS):
    """Stream Food-101 train split, take first `per_class` of each target class."""
    print(f"loading food101 (streaming)...")
    ds = load_dataset("food101", split="train", streaming=True)
    class_names = ds.features["label"].names
    target_idxs = {class_names.index(c): i for i, c in enumerate(FOOD_CLASSES)}
    print(f"target indices in food101: {target_idxs}")

    by_class = {i: [] for i in range(len(FOOD_CLASSES))}
    for ex in ds:
        cls_idx = ex["label"]
        if cls_idx not in target_idxs:
            continue
        local = target_idxs[cls_idx]
        if len(by_class[local]) < per_class:
            by_class[local].append(ex["image"].convert("RGB"))
        if all(len(v) >= per_class for v in by_class.values()):
            break

    pils, labels = [], []
    for cls in range(len(FOOD_CLASSES)):
        pils.extend(by_class[cls])
        labels.extend([cls] * len(by_class[cls]))

    print(f"  collected {len(pils)} images across {len(FOOD_CLASSES)} classes "
          f"({[len(by_class[i]) for i in range(len(FOOD_CLASSES))]})")
    return pils, np.array(labels)


def encode_with_clip_base_and_ft(model_name, adapter_dir, pils):
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
    pils, labels = sample_food()
    np.save(os.path.join(SCRIPT_DIR, "food_labels.npy"), labels)

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

        np.save(os.path.join(SCRIPT_DIR, f"food_base_{tag}.npy"), base_e)
        np.save(os.path.join(SCRIPT_DIR, f"food_ft_{tag}.npy"),   ft_e)
        print(f"  saved food_base_{tag}.npy  ({base_e.shape})")
        print(f"  saved food_ft_{tag}.npy    ({ft_e.shape})")

    print("\ndone. run visualize.py next.")
