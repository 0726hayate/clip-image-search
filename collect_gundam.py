import os
import json
import random
import requests
from io import BytesIO
from PIL import Image

# ── series wiki page titles (MediaWiki API titles, not URLs) ─────────────────
# these are the exact page names on gundam.fandom.com
SERIES_PAGES = {
    "uc":   "Mobile_Suit_Gundam_Unicorn",
    "age":  "Mobile_Suit_Gundam_AGE",
    "wfm":  "Mobile_Suit_Gundam_the_Witch_from_Mercury",
    "00":   "Mobile_Suit_Gundam_00",
    "ibo":  "Mobile_Suit_Gundam_IRON-BLOODED_ORPHANS",
    "grg":  "Gundam_Reconguista_in_G",
}

# ── output paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR  = os.path.join(SCRIPT_DIR, "gundam", "images")
LABELS_PATH = os.path.join(SCRIPT_DIR, "gundam", "labels.json")
TRAIN_PATH  = os.path.join(SCRIPT_DIR, "gundam", "train_files.json")
VAL_PATH    = os.path.join(SCRIPT_DIR, "gundam", "val_files.json")

WIKI_API    = "https://gundam.fandom.com/api.php"
HEADERS     = {"User-Agent": "Mozilla/5.0 (academic research)"}
TRAIN_RATIO = 0.8   # 80% train, 20% val
BATCH_SIZE  = 50    # max titles per imageinfo API call

# skip images whose filename suggests they're UI/logo/icon rather than mobile suit art
SKIP_KEYWORDS = ["logo", "icon", "banner", "wordmark", "favicon", "placeholder",
                 "wiki", "noimage", "button", "badge", "arrow", "bullet"]


def get_page_image_titles(page_title):
    """Return all image file titles listed on a wiki page, handling pagination."""
    titles = []
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "images",
        "imlimit": 500,       # max per request
        "format": "json",
    }
    while True:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15).json()
        for page in r.get("query", {}).get("pages", {}).values():
            titles += [img["title"] for img in page.get("images", [])]
        # if there are more images the API returns a 'continue' token
        if "continue" not in r:
            break
        params["imcontinue"] = r["continue"]["imcontinue"]
    return titles


def resolve_image_urls(file_titles):
    """Batch-resolve file titles to their direct image URLs using imageinfo."""
    urls = {}
    for i in range(0, len(file_titles), BATCH_SIZE):
        batch = "|".join(file_titles[i : i + BATCH_SIZE])
        params = {
            "action": "query",
            "titles": batch,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15).json()
        for page in r.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            info = page.get("imageinfo", [{}])
            if info and info[0].get("url"):
                urls[title] = info[0]["url"]
    return urls


def is_valid_file(title):
    """Return True if the file title looks like mobile suit art, not site chrome."""
    name_lower = title.lower()
    # must be an image format
    if not any(name_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"]):
        return False
    # skip UI elements
    if any(kw in name_lower for kw in SKIP_KEYWORDS):
        return False
    return True


def download_image(url, dest_path):
    """Download image from URL, save as RGB JPEG. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        # skip very small images — likely icons or thumbnails not worth training on
        if img.width < 100 or img.height < 100:
            return False
        img.save(dest_path, "JPEG")
        return True
    except Exception:
        return False


def collect_series(series_name, page_title):
    """Download all valid mobile suit images from one series wiki page."""
    out_dir = os.path.join(IMAGES_DIR, series_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [{series_name}] fetching image list from '{page_title}'...")
    all_titles = get_page_image_titles(page_title)

    # filter to plausible mobile suit images
    valid_titles = [t for t in all_titles if is_valid_file(t)]
    print(f"  [{series_name}] {len(all_titles)} files on page -> {len(valid_titles)} after filtering")

    # resolve file titles to direct image URLs
    url_map = resolve_image_urls(valid_titles)

    saved = 0
    for file_title, img_url in url_map.items():
        dest = os.path.join(out_dir, f"{saved}.jpg")
        if os.path.exists(dest):
            saved += 1
            continue   # already downloaded — skip on re-run
        if download_image(img_url, dest):
            saved += 1

    print(f"  [{series_name}] saved {saved} images")
    return saved


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)

    all_files = []   # list of (relative_path, series_label) tuples
    counts = {}

    for series_name, page_title in SERIES_PAGES.items():
        n = collect_series(series_name, page_title)
        counts[series_name] = n

        # record each saved file with its label
        series_dir = os.path.join(IMAGES_DIR, series_name)
        for fname in sorted(os.listdir(series_dir)):
            if fname.endswith(".jpg"):
                rel_path = os.path.join("gundam", "images", series_name, fname)
                all_files.append((rel_path, series_name))

    # save filename -> label mapping for finetune.py to consume
    labels = {path: label for path, label in all_files}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    # 80/20 split — shuffle first so val covers all series proportionally
    random.seed(42)
    random.shuffle(all_files)
    split = int(len(all_files) * TRAIN_RATIO)
    train_files = [p for p, _ in all_files[:split]]
    val_files   = [p for p, _ in all_files[split:]]

    with open(TRAIN_PATH, "w") as f:
        json.dump(train_files, f)
    with open(VAL_PATH, "w") as f:
        json.dump(val_files, f)

    print(f"\nper-series counts: {counts}")
    print(f"total: {len(all_files)} ({len(train_files)} train / {len(val_files)} val)")
    print("done. run finetune.py next.")
