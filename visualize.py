import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR      = os.path.join(SCRIPT_DIR, "plots")
FLICKR_CAT_PATH = os.path.join(SCRIPT_DIR, "flickr_categories.npy")

# per-model embedding files (produced by embed.py and finetune.py)
MODELS = {
    "b32":  "CLIP ViT-B/32\n(baseline)",
    "l14":  "CLIP ViT-L/14\n(exp1: better vision)",
    "jina": "Jina CLIP v2\n(exp2: better text)",
    "h14":  "OpenCLIP ViT-H/14\n(exp3: both scaled)",
}

# ── label scheme ──────────────────────────────────────────────────────────────
# Labels 0..6  → Flickr30K categories (assigned by CLIP zero-shot in categorize_flickr.py)
# Labels 7..12 → Gundam series (original 0..5 shifted by N_FLICKR_CATS=7)
N_FLICKR_CATS = 7

LABEL_NAMES = {
    # Flickr categories — pastel colours, smaller dots
    0: "Flickr: people",
    1: "Flickr: animals",
    2: "Flickr: sports",
    3: "Flickr: nature",
    4: "Flickr: food",
    5: "Flickr: vehicles",
    6: "Flickr: architecture",
    # Gundam series — saturated colours, larger dots
    7: "UC Unicorn",
    8: "AGE",
    9: "SEED",
    10: "Gundam 00",
    11: "IBO",
    12: "G-Reco",
}

# Flickr: pastel palette, distinct from the saturated Gundam colours
# Gundam: same saturated palette as before
COLORS = {
    0:  "lightpink",       # people
    1:  "burlywood",       # animals
    2:  "powderblue",      # sports
    3:  "palegreen",       # nature
    4:  "thistle",         # food
    5:  "lightyellow",     # vehicles
    6:  "lightgray",       # architecture
    7:  "steelblue",       # UC Unicorn
    8:  "darkorange",      # AGE
    9:  "tomato",          # SEED
    10: "gold",            # Gundam 00
    11: "seagreen",        # IBO
    12: "mediumpurple",    # G-Reco
}

# Flickr: smaller, more transparent; Gundam: larger, more opaque
SIZES = {
    0: 8,  1: 8,  2: 8,  3: 8,  4: 8,  5: 8,  6: 8,
    7: 22, 8: 22, 9: 22, 10: 22, 11: 22, 12: 22,
}
ALPHAS = {
    0: 0.45, 1: 0.45, 2: 0.45, 3: 0.45, 4: 0.45, 5: 0.45, 6: 0.45,
    7: 0.88, 8: 0.88, 9: 0.88, 10: 0.88, 11: 0.88, 12: 0.88,
}


def run_pca_tsne(all_embs, n_components_target=0.95, tag=""):
    """PCA (k chosen by explained variance >= threshold) then t-SNE to 2D.

    k and perplexity are determined systematically from the data — no hardcoding.
    Returns (2D coords, chosen k).
    """
    D = all_embs.shape[1]
    print(f"  [{tag}] PCA on {all_embs.shape} (input dim={D})...")
    pca_probe = PCA()
    pca_probe.fit(all_embs)

    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    # smallest k such that cumulative variance crosses the target
    k = int(np.searchsorted(cumvar, n_components_target)) + 1
    print(f"  [{tag}] k={k} explains {n_components_target*100:.0f}% variance")

    reduced = PCA(n_components=k).fit_transform(all_embs)

    N = len(reduced)
    perplexity = int(np.sqrt(N))   # principled default: roughly sqrt(N) neighbors
    print(f"  [{tag}] t-SNE: N={N}, perplexity={perplexity}")

    tsne      = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    coords_2d = tsne.fit_transform(reduced)

    return coords_2d, k


def make_scatter(ax, coords, all_labels, title):
    """Draw a scatter plot of embeddings colored by label."""
    for label_id in sorted(set(all_labels)):
        mask = all_labels == label_id
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[label_id],
            s=SIZES[label_id],
            alpha=ALPHAS[label_id],
            label=LABEL_NAMES[label_id],
            edgecolors="none",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── load Flickr category labels ───────────────────────────────────────
    if not os.path.exists(FLICKR_CAT_PATH):
        print(f"flickr_categories.npy not found. run categorize_flickr.py first.")
        raise SystemExit(1)
    flickr_cat_labels = np.load(FLICKR_CAT_PATH)   # (1000,) values 0..6

    # ── discover which models have all required embedding files ───────────
    available = []
    for tag in MODELS:
        img_path    = os.path.join(SCRIPT_DIR, f"img_embeddings_{tag}.npy")
        base_path   = os.path.join(SCRIPT_DIR, f"gundam_val_base_{tag}.npy")
        ft_path     = os.path.join(SCRIPT_DIR, f"gundam_val_ft_{tag}.npy")
        labels_path = os.path.join(SCRIPT_DIR, "gundam_val_labels.npy")
        if all(os.path.exists(p) for p in [img_path, base_path, ft_path, labels_path]):
            available.append(tag)
        else:
            missing = [os.path.basename(p) for p in [img_path, base_path, ft_path, labels_path]
                       if not os.path.exists(p)]
            print(f"[{tag}] missing: {missing}, skipping")

    if not available:
        print("no models with complete embeddings. run embed.py and finetune.py first.")
        raise SystemExit(1)

    # Gundam raw labels are 0..5; shift by N_FLICKR_CATS so they become 7..12
    gundam_labels_raw     = np.load(os.path.join(SCRIPT_DIR, "gundam_val_labels.npy"))
    gundam_labels_shifted = gundam_labels_raw + N_FLICKR_CATS

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # keep consistent 2D indexing

    for row, tag in enumerate(available):
        print(f"\n[{tag}] {MODELS[tag].strip()}")

        flickr_embs = np.load(os.path.join(SCRIPT_DIR, f"img_embeddings_{tag}.npy"))
        gundam_base = np.load(os.path.join(SCRIPT_DIR, f"gundam_val_base_{tag}.npy"))
        gundam_ft   = np.load(os.path.join(SCRIPT_DIR, f"gundam_val_ft_{tag}.npy"))

        N_flickr   = len(flickr_embs)
        # Flickr labels are the CLIP-assigned category (0..6)
        # Gundam labels are shifted to 7..12
        all_labels = np.concatenate([
            flickr_cat_labels,
            gundam_labels_shifted,
        ])

        all_base = np.vstack([flickr_embs, gundam_base])
        all_ft   = np.vstack([flickr_embs, gundam_ft])

        coords_base, _ = run_pca_tsne(all_base, tag=f"{tag}_base")
        coords_ft,   _ = run_pca_tsne(all_ft,   tag=f"{tag}_ft")

        model_label = MODELS[tag].replace("\n", " ")
        make_scatter(axes[row, 0], coords_base, all_labels,
                     f"{model_label} — zero-shot")
        make_scatter(axes[row, 1], coords_ft,   all_labels,
                     f"{model_label} — LoRA fine-tuned")

    # shared legend: Flickr categories first, then Gundam series
    handles = [
        mpatches.Patch(color=COLORS[i], label=LABEL_NAMES[i])
        for i in sorted(LABEL_NAMES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle(
        "Embedding space: Flickr30K categories (pastel) + Gundam series (saturated)\n"
        "Each row: zero-shot vs LoRA fine-tuned — PCA (95% var) → t-SNE (perplexity=√N)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "tsne_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved -> {out_path}")
    print("done.")
