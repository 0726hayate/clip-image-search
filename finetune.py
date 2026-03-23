import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image

# ── model selection ────────────────────────────────────────────────────────
# change MODEL_TAG to run a different model, or pass as first CLI arg:
#   "b32"  → CLIP ViT-B/32     (baseline)
#   "l14"  → CLIP ViT-L/14     (exp1: better vision)
#   "jina" → Jina CLIP v2      (exp2: better text — XLM-RoBERTa-large)
#   "h14"  → OpenCLIP ViT-H/14 (exp3: both scaled)
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "l14"

MODEL_CONFIGS = {
    "b32":  {"name": "openai/clip-vit-base-patch32",           "api": "clip"},
    "l14":  {"name": "openai/clip-vit-large-patch14",          "api": "clip"},
    "jina": {"name": "jinaai/jina-clip-v2",                    "api": "jina"},
    "h14":  {"name": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "api": "clip"},
}

MODEL_NAME = MODEL_CONFIGS[MODEL_TAG]["name"]
MODEL_API  = MODEL_CONFIGS[MODEL_TAG]["api"]

# ── paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH  = os.path.join(SCRIPT_DIR, "gundam", "labels.json")
TRAIN_PATH   = os.path.join(SCRIPT_DIR, "gundam", "train_files.json")
VAL_PATH     = os.path.join(SCRIPT_DIR, "gundam", "val_files.json")
ADAPTER_DIR  = os.path.join(SCRIPT_DIR, f"gundam_lora_{MODEL_TAG}")
VAL_BASE_PATH   = os.path.join(SCRIPT_DIR, f"gundam_val_base_{MODEL_TAG}.npy")
VAL_FT_PATH     = os.path.join(SCRIPT_DIR, f"gundam_val_ft_{MODEL_TAG}.npy")
VAL_LABELS_PATH = os.path.join(SCRIPT_DIR, "gundam_val_labels.npy")   # same across models

# ── training settings ─────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# Jina EVA02-L processes 512×512 images (1340 tokens) — backprop needs ~8× more VRAM than 224px models
BATCH_SIZE  = 8 if MODEL_TAG == "jina" else 32
N_EPOCHS    = 10
LR          = 2e-4
TEMPERATURE = 0.07

SERIES_TO_IDX = {"uc": 0, "age": 1, "seed": 2, "00": 3, "ibo": 4, "grg": 5}
IDX_TO_SERIES = {v: k for k, v in SERIES_TO_IDX.items()}


class GundamDataset(Dataset):
    """Loads Gundam images and returns (PIL image, series label index)."""

    def __init__(self, file_list, labels_map):
        self.file_list  = file_list
        self.labels_map = labels_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rel_path  = self.file_list[idx]
        full_path = os.path.join(SCRIPT_DIR, rel_path)
        img       = Image.open(full_path).convert("RGB")
        series    = self.labels_map[rel_path]
        label     = SERIES_TO_IDX[series]
        return img, label


def collate_clip(batch, processor):
    """Collate (PIL image, label) list into CLIPProcessor tensors."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"], labels


def supcon_loss(embs, labels, temperature=TEMPERATURE):
    """Supervised contrastive loss (Khosla et al. 2020).

    All same-series images in a batch are positives; cross-series are negatives.
    """
    B = embs.shape[0]
    sim = embs @ embs.T / temperature

    # -1e9 instead of -inf avoids nan in the log-sum-exp when pos_mask has zeros
    diag_mask = torch.eye(B, dtype=torch.bool, device=embs.device)
    sim       = sim.masked_fill(diag_mask, -1e9)

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # torch.where avoids -inf * 0 = nan at negative positions
    masked = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    loss   = -masked.sum(1) / pos_mask.sum(1).clamp(min=1)
    return loss.mean()


def encode_images_clip(model, processor, file_list, labels_map, batch_size=64):
    """Encode images with a CLIPModel, return (embeddings, labels)."""
    model.eval()
    all_embs, all_labels = [], []

    for i in range(0, len(file_list), batch_size):
        batch_paths = file_list[i : i + batch_size]
        images = []
        for rel_path in batch_paths:
            images.append(Image.open(os.path.join(SCRIPT_DIR, rel_path)).convert("RGB"))
            all_labels.append(SERIES_TO_IDX[labels_map[rel_path]])

        inputs      = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=pixel_values)
            # projection_dim is read from model.config — never hardcoded
            embs = model.visual_projection(vision_out.pooler_output)
            embs = F.normalize(embs, dim=-1)

        all_embs.append(embs.cpu().numpy())

    return np.vstack(all_embs), np.array(all_labels)


def encode_images_jina(model, file_list, labels_map, batch_size=32):
    """Encode images with Jina CLIP v2 (encode_image method)."""
    model.eval()
    all_embs, all_labels = [], []

    for i in range(0, len(file_list), batch_size):
        batch_paths = file_list[i : i + batch_size]
        images = []
        for rel_path in batch_paths:
            images.append(Image.open(os.path.join(SCRIPT_DIR, rel_path)).convert("RGB"))
            all_labels.append(SERIES_TO_IDX[labels_map[rel_path]])

        with torch.no_grad():
            embs = model.encode_image(images)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().numpy()
        all_embs.append(np.array(embs))

    return np.vstack(all_embs), np.array(all_labels)


def recall_at_k_series(embs, labels, ks=(1, 5, 10)):
    """Series Recall@K: for each image, is any same-series image in the top K?"""
    sim = embs @ embs.T
    np.fill_diagonal(sim, -1.0)

    hits = {k: 0 for k in ks}
    for i in range(len(embs)):
        same_series = set(np.where(labels == labels[i])[0].tolist()) - {i}
        if not same_series:
            continue
        ranked = np.argsort(-sim[i])
        for k in ks:
            if len(same_series & set(ranked[:k].tolist())) > 0:
                hits[k] += 1

    return {k: hits[k] / len(embs) * 100 for k in ks}


if __name__ == "__main__":
    with open(LABELS_PATH) as f: labels_map  = json.load(f)
    with open(TRAIN_PATH)  as f: train_files = json.load(f)
    with open(VAL_PATH)    as f: val_files   = json.load(f)

    print(f"model tag: {MODEL_TAG} | api: {MODEL_API}")
    print(f"train: {len(train_files)} images, val: {len(val_files)} images")
    print(f"\nloading {MODEL_NAME}...")

    if MODEL_API == "clip":
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        clip      = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

        # projection_dim is read from model.config — never hardcoded
        dim = clip.config.projection_dim
        print(f"projection_dim={dim} (from model.config)")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],   # standard in HF CLIPModel for all sizes
            lora_dropout=0.05,
            bias="none",
        )
        clip = get_peft_model(clip, lora_config)
        clip.print_trainable_parameters()

        # encode val set before fine-tuning
        print("\nencoding val set (base)...")
        val_embs_base, val_labels = encode_images_clip(clip, processor, val_files, labels_map)
        base_recall = recall_at_k_series(val_embs_base, val_labels)
        print(f"  base R@1={base_recall[1]:.1f}%  R@5={base_recall[5]:.1f}%  R@10={base_recall[10]:.1f}%")
        np.save(VAL_BASE_PATH, val_embs_base)
        np.save(VAL_LABELS_PATH, val_labels)

        # training
        train_dataset = GundamDataset(train_files, labels_map)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda b: collate_clip(b, processor),
        )
        optimizer = torch.optim.AdamW(
            [p for p in clip.parameters() if p.requires_grad], lr=LR
        )

        print(f"\nfine-tuning for {N_EPOCHS} epochs...")
        clip.train()
        for epoch in range(N_EPOCHS):
            total_loss = 0.0
            for pixel_values, labels_batch in train_loader:
                pixel_values = pixel_values.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)

                vision_out = clip.vision_model(pixel_values=pixel_values)
                embs       = clip.visual_projection(vision_out.pooler_output)
                embs       = F.normalize(embs, dim=-1)

                loss = supcon_loss(embs, labels_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  epoch {epoch+1}/{N_EPOCHS}  loss={total_loss/len(train_loader):.4f}")

        # encode val set after fine-tuning
        print("\nencoding val set (fine-tuned)...")
        val_embs_ft, _ = encode_images_clip(clip, processor, val_files, labels_map)
        ft_recall = recall_at_k_series(val_embs_ft, val_labels)
        np.save(VAL_FT_PATH, val_embs_ft)

        os.makedirs(ADAPTER_DIR, exist_ok=True)
        clip.save_pretrained(ADAPTER_DIR)
        print(f"\nLoRA adapter saved to {ADAPTER_DIR}/")

    else:  # jina
        clip = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        clip.eval()

        # EVA02 attention uses f.linear(weight=self.q_proj.weight) — bypasses module call.
        # Target attn.proj (output projection) and mlp.fc1/fc2 which ARE called as modules.
        lora_targets = [
            name for name, mod in clip.named_modules()
            if "vision_model.blocks" in name
            and isinstance(mod, torch.nn.Linear)
            and name.split(".")[-1] in ("proj", "fc1", "fc2")
        ]
        print(f"\nLoRA target modules ({len(lora_targets)} total, first 4):")
        for t in lora_targets[:4]:
            print(f"  {t}")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=lora_targets,
            lora_dropout=0.05,
            bias="none",
        )
        clip = get_peft_model(clip, lora_config)
        clip.print_trainable_parameters()

        # encode val set before fine-tuning
        print("\nencoding val set (base)...")
        val_embs_base, val_labels = encode_images_jina(clip, val_files, labels_map)
        base_recall = recall_at_k_series(val_embs_base, val_labels)
        print(f"  base R@1={base_recall[1]:.1f}%  R@5={base_recall[5]:.1f}%  R@10={base_recall[10]:.1f}%")
        np.save(VAL_BASE_PATH, val_embs_base)
        np.save(VAL_LABELS_PATH, val_labels)

        # training with Jina — use get_image_features() directly (no inference_mode)
        # encode_image() has @torch.inference_mode() so gradients are blocked there
        jina_preprocess = clip.get_preprocess()

        def collate_jina(batch):
            images = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            pixel_values = jina_preprocess(images)
            return pixel_values, labels

        train_dataset = GundamDataset(train_files, labels_map)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_jina,
        )
        optimizer = torch.optim.AdamW(
            [p for p in clip.parameters() if p.requires_grad], lr=LR
        )

        print(f"\nfine-tuning for {N_EPOCHS} epochs...")
        clip.train()
        for epoch in range(N_EPOCHS):
            total_loss = 0.0
            for pixel_values, labels_batch in train_loader:
                pixel_values  = pixel_values.to(DEVICE)
                labels_batch  = labels_batch.to(DEVICE)

                embs = clip.get_image_features(pixel_values)
                embs = F.normalize(embs.float(), dim=-1)

                loss = supcon_loss(embs, labels_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  epoch {epoch+1}/{N_EPOCHS}  loss={total_loss/len(train_loader):.4f}")

        # encode val set after fine-tuning
        print("\nencoding val set (fine-tuned)...")
        val_embs_ft, _ = encode_images_jina(clip, val_files, labels_map)
        ft_recall = recall_at_k_series(val_embs_ft, val_labels)
        np.save(VAL_FT_PATH, val_embs_ft)

        os.makedirs(ADAPTER_DIR, exist_ok=True)
        clip.save_pretrained(ADAPTER_DIR)
        print(f"\nLoRA adapter saved to {ADAPTER_DIR}/")

    # ── results ──────────────────────────────────────────────────────────
    ks = (1, 5, 10)
    print("\n" + "=" * 52)
    print(f"  {MODEL_TAG} | {MODEL_NAME}")
    print(f"  {'model':<22} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print("-" * 52)
    print(f"  {'base CLIP':<22} {base_recall[1]:>6.1f}% {base_recall[5]:>6.1f}% {base_recall[10]:>6.1f}%")
    print(f"  {'+ LoRA':<22} {ft_recall[1]:>6.1f}% {ft_recall[5]:>6.1f}% {ft_recall[10]:>6.1f}%")
    print("=" * 52)
    print("\nseries Recall@K: given a mobile suit image, find same-series images.")
    print("done. run visualize.py next.")
