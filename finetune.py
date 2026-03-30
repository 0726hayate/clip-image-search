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

# ── model + dataset selection ─────────────────────────────────────────────
# CLI: python finetune.py <model_tag> [<dataset_tag>]
#   model_tag:   "b32" | "l14" | "jina" | "h14"  (default "l14")
#   dataset_tag: "gundam" | "pokemon" | "paintings"  (default "gundam")
MODEL_TAG   = sys.argv[1] if len(sys.argv) > 1 else "l14"
DATASET_TAG = sys.argv[2] if len(sys.argv) > 2 else "gundam"

MODEL_CONFIGS = {
    "b32":  {"name": "openai/clip-vit-base-patch32",           "api": "clip"},
    "l14":  {"name": "openai/clip-vit-large-patch14",          "api": "clip"},
    "jina": {"name": "jinaai/jina-clip-v2",                    "api": "jina"},
    "h14":  {"name": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "api": "clip"},
}

MODEL_NAME = MODEL_CONFIGS[MODEL_TAG]["name"]
MODEL_API  = MODEL_CONFIGS[MODEL_TAG]["api"]

# ── paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH     = os.path.join(SCRIPT_DIR, DATASET_TAG, "labels.json")
TRAIN_PATH      = os.path.join(SCRIPT_DIR, DATASET_TAG, "train_files.json")
VAL_PATH        = os.path.join(SCRIPT_DIR, DATASET_TAG, "val_files.json")
ADAPTER_DIR     = os.path.join(SCRIPT_DIR, f"{DATASET_TAG}_lora_{MODEL_TAG}")
VAL_BASE_PATH   = os.path.join(SCRIPT_DIR, f"{DATASET_TAG}_val_base_{MODEL_TAG}.npy")
VAL_FT_PATH     = os.path.join(SCRIPT_DIR, f"{DATASET_TAG}_val_ft_{MODEL_TAG}.npy")
VAL_LABELS_PATH = os.path.join(SCRIPT_DIR, f"{DATASET_TAG}_val_labels.npy")

# ── training settings ─────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# Jina EVA02-L processes 512×512 images (1340 tokens) — backprop needs ~8× more VRAM than 224px models
BATCH_SIZE  = 8 if MODEL_TAG == "jina" else 32
N_EPOCHS    = 10
LR          = 2e-4
TEMPERATURE = 0.07


def build_class_idx(labels_map):
    """Compute deterministic class -> index map from labels.json (sorted order)."""
    class_names = sorted(set(labels_map.values()))
    return {c: i for i, c in enumerate(class_names)}


class NicheDataset(Dataset):
    """Returns (PIL image, class index) for any niche dataset following the same layout."""

    def __init__(self, file_list, labels_map, class_to_idx):
        self.file_list    = file_list
        self.labels_map   = labels_map
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rel_path  = self.file_list[idx]
        full_path = os.path.join(SCRIPT_DIR, rel_path)
        img       = Image.open(full_path).convert("RGB")
        cls_name  = self.labels_map[rel_path]
        return img, self.class_to_idx[cls_name]


def collate_clip(batch, processor):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"], labels


def supcon_loss(embs, labels, temperature=TEMPERATURE):
    """Supervised contrastive loss (Khosla et al. 2020)."""
    B = embs.shape[0]
    sim = embs @ embs.T / temperature
    diag_mask = torch.eye(B, dtype=torch.bool, device=embs.device)
    sim       = sim.masked_fill(diag_mask, -1e9)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~diag_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    masked = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    loss   = -masked.sum(1) / pos_mask.sum(1).clamp(min=1)
    return loss.mean()


def _load_pils(items):
    """Accept either file paths (str) or already-loaded PIL images. Returns list of RGB PILs."""
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(Image.open(os.path.join(SCRIPT_DIR, item)).convert("RGB"))
        else:
            out.append(item.convert("RGB") if item.mode != "RGB" else item)
    return out


def encode_images_clip(model, processor, items, labels_map=None, class_to_idx=None,
                       batch_size=64):
    """Encode images with a CLIPModel.

    items: list of file paths (str) OR list of PIL images.
    If items are paths AND labels_map+class_to_idx provided, also returns labels.
    """
    model.eval()
    all_embs, all_labels = [], []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        images = _load_pils(batch)
        if labels_map is not None and class_to_idx is not None and isinstance(batch[0], str):
            for rel_path in batch:
                all_labels.append(class_to_idx[labels_map[rel_path]])

        inputs       = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=pixel_values)
            embs = model.visual_projection(vision_out.pooler_output)
            embs = F.normalize(embs, dim=-1)

        all_embs.append(embs.cpu().numpy())

    embs_arr = np.vstack(all_embs)
    if all_labels:
        return embs_arr, np.array(all_labels)
    return embs_arr, None


def encode_images_jina(model, items, labels_map=None, class_to_idx=None, batch_size=32):
    """Encode images with Jina CLIP v2.

    items: list of file paths (str) OR list of PIL images.
    """
    model.eval()
    all_embs, all_labels = [], []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        images = _load_pils(batch)
        if labels_map is not None and class_to_idx is not None and isinstance(batch[0], str):
            for rel_path in batch:
                all_labels.append(class_to_idx[labels_map[rel_path]])

        with torch.no_grad():
            embs = model.encode_image(images)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().numpy()
        all_embs.append(np.array(embs))

    embs_arr = np.vstack(all_embs)
    if all_labels:
        return embs_arr, np.array(all_labels)
    return embs_arr, None


def recall_at_k_series(embs, labels, ks=(1, 5, 10)):
    """For each image, is any same-class image in the top K?"""
    sim = embs @ embs.T
    np.fill_diagonal(sim, -1.0)

    hits = {k: 0 for k in ks}
    for i in range(len(embs)):
        same = set(np.where(labels == labels[i])[0].tolist()) - {i}
        if not same:
            continue
        ranked = np.argsort(-sim[i])
        for k in ks:
            if len(same & set(ranked[:k].tolist())) > 0:
                hits[k] += 1

    return {k: hits[k] / len(embs) * 100 for k in ks}


if __name__ == "__main__":
    with open(LABELS_PATH) as f: labels_map  = json.load(f)
    with open(TRAIN_PATH)  as f: train_files = json.load(f)
    with open(VAL_PATH)    as f: val_files   = json.load(f)

    class_to_idx = build_class_idx(labels_map)
    print(f"model: {MODEL_TAG} ({MODEL_API}) | dataset: {DATASET_TAG} | classes: {list(class_to_idx)}")
    print(f"train: {len(train_files)} | val: {len(val_files)}")
    print(f"\nloading {MODEL_NAME}...")

    if MODEL_API == "clip":
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        clip      = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

        dim = clip.config.projection_dim
        print(f"projection_dim={dim} (from model.config)")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        clip = get_peft_model(clip, lora_config)
        clip.print_trainable_parameters()

        print("\nencoding val set (base)...")
        val_embs_base, val_labels = encode_images_clip(
            clip, processor, val_files, labels_map, class_to_idx)
        base_recall = recall_at_k_series(val_embs_base, val_labels)
        print(f"  base R@1={base_recall[1]:.1f}%  R@5={base_recall[5]:.1f}%  R@10={base_recall[10]:.1f}%")
        np.save(VAL_BASE_PATH, val_embs_base)
        np.save(VAL_LABELS_PATH, val_labels)

        train_dataset = NicheDataset(train_files, labels_map, class_to_idx)
        train_loader  = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=lambda b: collate_clip(b, processor),
        )
        optimizer = torch.optim.AdamW(
            [p for p in clip.parameters() if p.requires_grad], lr=LR)

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

        print("\nencoding val set (fine-tuned)...")
        val_embs_ft, _ = encode_images_clip(
            clip, processor, val_files, labels_map, class_to_idx)
        ft_recall = recall_at_k_series(val_embs_ft, val_labels)
        np.save(VAL_FT_PATH, val_embs_ft)

        os.makedirs(ADAPTER_DIR, exist_ok=True)
        clip.save_pretrained(ADAPTER_DIR)
        print(f"\nLoRA adapter saved to {ADAPTER_DIR}/")

    else:  # jina
        clip = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
        clip.eval()

        # EVA02 attention uses f.linear(weight=self.q_proj.weight) — bypasses module call.
        # Target attn.proj (output projection) which IS called as self.proj(x).
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
            r=16, lora_alpha=32, target_modules=lora_targets,
            lora_dropout=0.05, bias="none",
        )
        clip = get_peft_model(clip, lora_config)
        clip.print_trainable_parameters()

        print("\nencoding val set (base)...")
        val_embs_base, val_labels = encode_images_jina(
            clip, val_files, labels_map, class_to_idx)
        base_recall = recall_at_k_series(val_embs_base, val_labels)
        print(f"  base R@1={base_recall[1]:.1f}%  R@5={base_recall[5]:.1f}%  R@10={base_recall[10]:.1f}%")
        np.save(VAL_BASE_PATH, val_embs_base)
        np.save(VAL_LABELS_PATH, val_labels)

        # training: bypass encode_image (has @torch.inference_mode); use get_image_features
        jina_preprocess = clip.get_preprocess()

        def collate_jina(batch):
            images = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            pixel_values = jina_preprocess(images)
            return pixel_values, labels

        train_dataset = NicheDataset(train_files, labels_map, class_to_idx)
        train_loader  = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_jina,
        )
        optimizer = torch.optim.AdamW(
            [p for p in clip.parameters() if p.requires_grad], lr=LR)

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

        print("\nencoding val set (fine-tuned)...")
        val_embs_ft, _ = encode_images_jina(
            clip, val_files, labels_map, class_to_idx)
        ft_recall = recall_at_k_series(val_embs_ft, val_labels)
        np.save(VAL_FT_PATH, val_embs_ft)

        os.makedirs(ADAPTER_DIR, exist_ok=True)
        clip.save_pretrained(ADAPTER_DIR)
        print(f"\nLoRA adapter saved to {ADAPTER_DIR}/")

    # ── results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print(f"  {MODEL_TAG} | {DATASET_TAG} | {MODEL_NAME}")
    print(f"  {'model':<22} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    print("-" * 56)
    print(f"  {'base CLIP':<22} {base_recall[1]:>6.1f}% {base_recall[5]:>6.1f}% {base_recall[10]:>6.1f}%")
    print(f"  {'+ LoRA':<22} {ft_recall[1]:>6.1f}% {ft_recall[5]:>6.1f}% {ft_recall[10]:>6.1f}%")
    print("=" * 56)
    print(f"\nclass Recall@K on {DATASET_TAG}: given an image, find same-class images.")
