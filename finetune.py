import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH  = os.path.join(SCRIPT_DIR, "gundam", "labels.json")
TRAIN_PATH   = os.path.join(SCRIPT_DIR, "gundam", "train_files.json")
VAL_PATH     = os.path.join(SCRIPT_DIR, "gundam", "val_files.json")
ADAPTER_DIR  = os.path.join(SCRIPT_DIR, "gundam_lora")

# ── model and training settings ───────────────────────────────────────────────
MODEL_NAME   = "openai/clip-vit-base-patch32"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 32
N_EPOCHS     = 10
LR           = 2e-4
TEMPERATURE  = 0.07   # same temperature CLIP was originally trained with

# 6 Gundam series, each gets an integer label for SupCon loss
SERIES_TO_IDX = {"uc": 0, "age": 1, "seed": 2, "00": 3, "ibo": 4, "grg": 5}
IDX_TO_SERIES = {v: k for k, v in SERIES_TO_IDX.items()}


class GundamDataset(Dataset):
    """Loads Gundam images and returns (PIL image, series label index)."""

    def __init__(self, file_list, labels_map):
        self.file_list = file_list
        self.labels_map = labels_map   # path -> series string

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rel_path = self.file_list[idx]
        full_path = os.path.join(SCRIPT_DIR, rel_path)
        img = Image.open(full_path).convert("RGB")
        series = self.labels_map[rel_path]
        label  = SERIES_TO_IDX[series]
        return img, label


def collate_fn(batch, processor):
    """Collate a batch of (PIL image, label) into processor-formatted tensors."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    # processor handles resizing, normalization, and pixel_values tensor creation
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"], labels


def supcon_loss(embs, labels, temperature=TEMPERATURE):
    """Supervised contrastive loss (Khosla et al. 2020).

    Same series images are positives for each other; cross-series are negatives.
    This is strictly stronger than vanilla InfoNCE which only has one positive
    per anchor — with multiple positives per class the signal is richer.
    """
    B = embs.shape[0]
    # cosine similarity matrix — embeddings are already L2-normalized
    sim = embs @ embs.T / temperature   # (B, B)

    # use -1e9 (not -inf) for the diagonal mask to avoid nan from -inf * 0
    diag_mask = torch.eye(B, dtype=torch.bool, device=embs.device)
    sim = sim.masked_fill(diag_mask, -1e9)

    # positive mask: True where two samples share the same series label (excluding self)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag_mask   # (B, B)

    # log-softmax over all non-self similarities
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # sum log-prob only at positive positions; use torch.where to avoid -inf * 0 = nan
    masked = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    # clamp(min=1) prevents div-by-zero when a batch has only one sample of a class
    loss = -masked.sum(1) / pos_mask.sum(1).clamp(min=1)
    return loss.mean()


if __name__ == "__main__":
    # load data
    with open(LABELS_PATH)  as f: labels_map  = json.load(f)
    with open(TRAIN_PATH)   as f: train_files = json.load(f)

    print(f"train: {len(train_files)} images")

    # load CLIP from transformers (not sentence-transformers) so peft can wrap it
    print(f"\nloading CLIP ({MODEL_NAME})...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    clip = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # apply LoRA to the vision encoder attention layers (q_proj and v_proj)
    # we only adapt the vision side — the text encoder already understands series names
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],   # standard LoRA targets in transformer attention
        lora_dropout=0.05,
        bias="none",
    )
    clip = get_peft_model(clip, lora_config)
    clip.print_trainable_parameters()   # should be ~1% of total params

    # ── training ─────────────────────────────────────────────────────────────
    train_dataset = GundamDataset(train_files, labels_map)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor),
    )

    optimizer = torch.optim.AdamW(
        [p for p in clip.parameters() if p.requires_grad],   # only LoRA params
        lr=LR,
    )

    print(f"\nfine-tuning for {N_EPOCHS} epochs...")
    clip.train()
    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        for pixel_values, labels_batch in train_loader:
            pixel_values = pixel_values.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            # forward pass through vision encoder only
            vision_out = clip.vision_model(pixel_values=pixel_values)
            embs = clip.visual_projection(vision_out.pooler_output)
            embs = F.normalize(embs, dim=-1)   # normalize before contrastive loss

            loss = supcon_loss(embs, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  epoch {epoch+1}/{N_EPOCHS}  loss={avg_loss:.4f}")

    # ── save LoRA adapter weights ─────────────────────────────────────────────
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    clip.save_pretrained(ADAPTER_DIR)
    print(f"\nLoRA adapter saved to {ADAPTER_DIR}/")
    print("done.")
