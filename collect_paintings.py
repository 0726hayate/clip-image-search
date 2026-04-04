import os
import json
import random
from datasets import load_dataset

# ── 8 art movements spanning medieval to modern ──────────────────────────────
# Picked for visual distinctness from each other (Renaissance flatness vs
# Baroque drama vs Impressionist brushwork vs Cubist fragmentation, etc).
# Note: Surrealism is NOT in huggan/wikiart's style taxonomy; using Pop_Art
# as the modern slot instead.
MOVEMENTS = [
    "Early_Renaissance",
    "Baroque",
    "Rococo",
    "Romanticism",
    "Impressionism",
    "Post_Impressionism",
    "Cubism",
    "Pop_Art",
]

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "paintings", "images")
LABELS_PATH = os.path.join(SCRIPT_DIR, "paintings", "labels.json")
TRAIN_PATH  = os.path.join(SCRIPT_DIR, "paintings", "train_files.json")
VAL_PATH    = os.path.join(SCRIPT_DIR, "paintings", "val_files.json")

PER_MOVEMENT_CAP = 60   # match the per-class budget used by collect_pokemon.py
TRAIN_RATIO      = 0.8


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("loading huggan/wikiart (streaming)...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    style_names = ds.features["style"].names
    target_idxs = {style_names.index(m): m for m in MOVEMENTS}
    counts = {m: 0 for m in MOVEMENTS}
    print(f"target style indices: {target_idxs}")

    # Iterate streaming and stop when every class hits its cap
    for i, sample in enumerate(ds):
        sidx = sample["style"]
        if sidx not in target_idxs:
            continue
        movement = target_idxs[sidx]
        if counts[movement] >= PER_MOVEMENT_CAP:
            continue

        out_dir = os.path.join(IMAGES_DIR, movement)
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, f"{counts[movement]:03d}.jpg")
        try:
            img = sample["image"].convert("RGB")
            if img.width < 100 or img.height < 100:
                continue
            img.save(dest, "JPEG", quality=92)
            counts[movement] += 1
        except Exception:
            continue

        if all(c >= PER_MOVEMENT_CAP for c in counts.values()):
            print(f"all classes filled at example {i}")
            break

        if i % 1000 == 0 and i > 0:
            print(f"  scanned {i} examples; counts so far: {counts}")

    # Build labels + splits
    all_files = []
    for movement in MOVEMENTS:
        mdir = os.path.join(IMAGES_DIR, movement)
        if not os.path.isdir(mdir):
            continue
        for fname in sorted(os.listdir(mdir)):
            if fname.endswith(".jpg"):
                rel_path = os.path.join("paintings", "images", movement, fname)
                all_files.append((rel_path, movement))

    labels = {p: m for p, m in all_files}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    random.seed(42)
    random.shuffle(all_files)
    split = int(len(all_files) * TRAIN_RATIO)
    train_files = [p for p, _ in all_files[:split]]
    val_files   = [p for p, _ in all_files[split:]]

    with open(TRAIN_PATH, "w") as f: json.dump(train_files, f)
    with open(VAL_PATH,   "w") as f: json.dump(val_files,   f)

    print(f"\nper-movement counts: {counts}")
    print(f"total: {len(all_files)} ({len(train_files)} train / {len(val_files)} val)")
    print("done. run finetune.py h14 paintings next.")


if __name__ == "__main__":
    main()
