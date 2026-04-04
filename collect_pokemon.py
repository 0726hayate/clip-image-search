import os
import json
import random
import requests
from io import BytesIO
from PIL import Image

# ── 8 main Pokémon types we'll specialize on ─────────────────────────────────
# Picked for visual distinctness and even species count. Skipping niche types
# (steel/poison/ice/etc.) to keep the per-class set clean.
TYPES = ["fire", "water", "grass", "electric", "psychic", "dark", "dragon", "fairy"]

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "pokemon", "images")
LABELS_PATH = os.path.join(SCRIPT_DIR, "pokemon", "labels.json")
TRAIN_PATH  = os.path.join(SCRIPT_DIR, "pokemon", "train_files.json")
VAL_PATH    = os.path.join(SCRIPT_DIR, "pokemon", "val_files.json")

POKEAPI       = "https://pokeapi.co/api/v2"
ARTWORK_URL   = ("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/"
                 "pokemon/other/official-artwork/{id}.png")
HEADERS       = {"User-Agent": "Mozilla/5.0 (academic research)"}
TRAIN_RATIO   = 0.8
PER_TYPE_CAP  = 60   # at most this many species per type to keep classes balanced


def get_species_for_type(type_name):
    """Return a list of (pokemon_id, name) for all species of this type.

    Each Pokémon may have 1-2 types. We accept a Pokémon for a type only if
    that type is its FIRST (slot 1) type — avoids duplicates across classes.
    """
    r = requests.get(f"{POKEAPI}/type/{type_name}", headers=HEADERS, timeout=15).json()
    out = []
    for entry in r.get("pokemon", []):
        if entry.get("slot") != 1:
            continue
        url = entry["pokemon"]["url"]   # .../pokemon/{id}/
        try:
            pid = int(url.rstrip("/").split("/")[-1])
        except ValueError:
            continue
        # PokeAPI species past id ~1025 may not have official artwork yet
        if pid > 1025:
            continue
        out.append((pid, entry["pokemon"]["name"]))
    return out


def download_artwork(pid, dest_path):
    """Download the official artwork PNG for a Pokémon id; returns True on success."""
    url = ARTWORK_URL.format(id=pid)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return False
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if img.width < 100 or img.height < 100:
            return False
        img.save(dest_path, "JPEG", quality=92)
        return True
    except Exception:
        return False


def collect_type(type_name):
    out_dir = os.path.join(IMAGES_DIR, type_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [{type_name}] fetching species list...")
    species = get_species_for_type(type_name)
    # cap per type so classes are balanced and contrastive batches are reasonable
    species = species[:PER_TYPE_CAP]

    saved = 0
    for pid, name in species:
        dest = os.path.join(out_dir, f"{pid:04d}_{name}.jpg")
        if os.path.exists(dest):
            saved += 1
            continue
        if download_artwork(pid, dest):
            saved += 1

    print(f"  [{type_name}] saved {saved}/{len(species)} species")
    return saved


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)

    counts = {}
    all_files = []
    for type_name in TYPES:
        n = collect_type(type_name)
        counts[type_name] = n

        type_dir = os.path.join(IMAGES_DIR, type_name)
        for fname in sorted(os.listdir(type_dir)):
            if fname.endswith(".jpg"):
                rel_path = os.path.join("pokemon", "images", type_name, fname)
                all_files.append((rel_path, type_name))

    labels = {p: t for p, t in all_files}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    random.seed(42)
    random.shuffle(all_files)
    split = int(len(all_files) * TRAIN_RATIO)
    train_files = [p for p, _ in all_files[:split]]
    val_files   = [p for p, _ in all_files[split:]]

    with open(TRAIN_PATH, "w") as f: json.dump(train_files, f)
    with open(VAL_PATH,   "w") as f: json.dump(val_files,   f)

    print(f"\nper-type counts: {counts}")
    print(f"total: {len(all_files)} ({len(train_files)} train / {len(val_files)} val)")
    print("done. run finetune.py h14 pokemon next.")
