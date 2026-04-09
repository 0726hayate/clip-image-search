"""Stream huggan/wikiart, save 60 paintings per movement, with artist tracking.

Why artist tracking matters: wikiart has many paintings per artist, and the
"style" label correlates with artist signature (Monet → Impressionism, etc).
A naive random train/val split puts the same artist in both, so the LoRA can
shortcut to "this brushstroke is Monet's" rather than learning movement-level
features. Capturing the artist lets finetune.py do an artist-disjoint split.

Outputs:
    paintings/images/{movement}/{idx}.jpg   image files
    paintings/labels.json                   {rel_path: movement}
    paintings/artists.json                  {rel_path: artist}    ← NEW
    paintings/all_files.json                [rel_path, ...] (canonical order)
    paintings/train_files.json              80/20 artist-disjoint train
    paintings/val_files.json                20% val (no artist overlap with train)
"""

import os
import json
import random
from collections import defaultdict
from datasets import load_dataset

# 8 art movements spanning medieval to modern. Surrealism is NOT in
# huggan/wikiart's style taxonomy; using Pop_Art as the modern slot instead.
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

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR   = os.path.join(SCRIPT_DIR, "paintings", "images")
LABELS_PATH  = os.path.join(SCRIPT_DIR, "paintings", "labels.json")
ARTISTS_PATH = os.path.join(SCRIPT_DIR, "paintings", "artists.json")
ALL_PATH     = os.path.join(SCRIPT_DIR, "paintings", "all_files.json")
TRAIN_PATH   = os.path.join(SCRIPT_DIR, "paintings", "train_files.json")
VAL_PATH     = os.path.join(SCRIPT_DIR, "paintings", "val_files.json")

PER_MOVEMENT_CAP   = 60       # target per-movement
# Diversity cap is genre-dependent: wikiart has ~10+ artists for major movements
# (Impressionism, Cubism) but only 2-3 for Pop_Art and Rococo. A flat low cap
# would deadlock on the sparse movements.
PER_ARTIST_CAP     = 30       # 2 artists × 30 = 60 minimum to fill any movement
SCAN_HARD_LIMIT    = 30000    # if not full by here, accept partial
TRAIN_RATIO        = 0.8
SEED               = 42


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Wipe any stale collection (the previous PER_MOVEMENT_CAP=60 run is invalid
    # for this artist-diversity-aware re-collection).
    import shutil
    if os.path.isdir(IMAGES_DIR):
        shutil.rmtree(IMAGES_DIR)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("loading huggan/wikiart (streaming)...")
    ds = load_dataset("huggan/wikiart", split="train", streaming=True)
    style_names  = ds.features["style"].names
    artist_names = ds.features["artist"].names
    target_idxs  = {style_names.index(m): m for m in MOVEMENTS}
    counts       = {m: 0 for m in MOVEMENTS}
    artist_count = {m: defaultdict(int) for m in MOVEMENTS}   # per-(movement, artist) image count
    artist_map   = {}                                          # rel_path → artist name
    print(f"target style indices: {target_idxs}")

    for i, sample in enumerate(ds):
        if i > SCAN_HARD_LIMIT:
            print(f"hit scan hard limit {SCAN_HARD_LIMIT}; stopping early")
            break

        sidx = sample["style"]
        if sidx not in target_idxs:
            continue
        movement = target_idxs[sidx]
        if counts[movement] >= PER_MOVEMENT_CAP:
            continue

        artist_idx  = sample["artist"]
        artist_name = artist_names[artist_idx] if artist_idx is not None else "unknown"
        if artist_count[movement][artist_name] >= PER_ARTIST_CAP:
            continue   # diversity cap — go find another artist for this movement

        out_dir = os.path.join(IMAGES_DIR, movement)
        os.makedirs(out_dir, exist_ok=True)
        dest_rel = os.path.join("paintings", "images", movement, f"{counts[movement]:03d}.jpg")
        dest_abs = os.path.join(SCRIPT_DIR, dest_rel)
        try:
            img = sample["image"].convert("RGB")
            if img.width < 100 or img.height < 100:
                continue
            img.save(dest_abs, "JPEG", quality=92)
            artist_map[dest_rel] = artist_name
            artist_count[movement][artist_name] += 1
            counts[movement] += 1
        except Exception:
            continue

        if all(c >= PER_MOVEMENT_CAP for c in counts.values()):
            print(f"all classes filled at example {i}")
            break

        if i % 5000 == 0 and i > 0:
            print(f"  scanned {i} examples; counts so far: {counts}")

    all_files = []
    for movement in MOVEMENTS:
        mdir = os.path.join(IMAGES_DIR, movement)
        if not os.path.isdir(mdir):
            continue
        for fname in sorted(os.listdir(mdir)):
            if fname.endswith(".jpg"):
                rel_path = os.path.join("paintings", "images", movement, fname)
                all_files.append((rel_path, movement))

    labels_map = {p: m for p, m in all_files}
    with open(LABELS_PATH, "w") as f:  json.dump(labels_map, f, indent=2)
    with open(ARTISTS_PATH, "w") as f: json.dump(artist_map, f, indent=2)
    with open(ALL_PATH, "w") as f:     json.dump([p for p, _ in all_files], f)

    # Artist-disjoint train/val split: per movement, partition the artist set
    # 80/20 (by artist count, not image count), then expand to file paths.
    rng = random.Random(SEED)
    train_files, val_files = [], []
    for movement in MOVEMENTS:
        files_in_m = [p for p, m in all_files if m == movement]
        artists_in_m = sorted(set(artist_map.get(p, "unknown") for p in files_in_m))
        rng.shuffle(artists_in_m)
        cut = max(1, int(len(artists_in_m) * TRAIN_RATIO))
        train_artists = set(artists_in_m[:cut])
        for p in files_in_m:
            if artist_map.get(p, "unknown") in train_artists:
                train_files.append(p)
            else:
                val_files.append(p)

    with open(TRAIN_PATH, "w") as f: json.dump(train_files, f)
    with open(VAL_PATH,   "w") as f: json.dump(val_files,   f)

    n_artists_total = len(set(artist_map.values()))
    print(f"\nper-movement counts: {counts}")
    print(f"total: {len(all_files)} images, {n_artists_total} unique artists")
    print(f"split: {len(train_files)} train / {len(val_files)} val (artist-disjoint per movement)")
    print("done. run finetune.py h14 paintings next (or with a fold idx).")


if __name__ == "__main__":
    main()
