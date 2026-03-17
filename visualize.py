import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
IMG_EMB_PATH    = os.path.join(SCRIPT_DIR, "img_embeddings.npy")     # flickr images
VAL_BASE_PATH   = os.path.join(SCRIPT_DIR, "gundam_val_base.npy")    # gundam base
VAL_FT_PATH     = os.path.join(SCRIPT_DIR, "gundam_val_ft.npy")      # gundam fine-tuned
VAL_LABELS_PATH = os.path.join(SCRIPT_DIR, "gundam_val_labels.npy")  # int labels 0–5
PLOTS_DIR       = os.path.join(SCRIPT_DIR, "plots")

# ── label names and colors ────────────────────────────────────────────────────
# label 0 = flickr (not a gundam series — used as background context)
# labels 1–6 = the 6 gundam series from finetune.py SERIES_TO_IDX
LABEL_NAMES = {
    0: "Flickr30K",
    1: "UC Unicorn",
    2: "AGE",
    3: "SEED",
    4: "Gundam 00",
    5: "IBO",
    6: "G-Reco",
}
COLORS = {
    0: "lightgrey",
    1: "steelblue",
    2: "darkorange",
    3: "tomato",
    4: "gold",
    5: "seagreen",
    6: "mediumpurple",
}
SIZES = {0: 6, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20}
ALPHAS = {0: 0.3, 1: 0.85, 2: 0.85, 3: 0.85, 4: 0.85, 5: 0.85, 6: 0.85}


def run_pca_tsne(all_embs, n_components_target=0.95):
    """PCA with systematic k (explained variance >= threshold) then t-SNE to 2D.

    Returns (2D coords, chosen k) and saves the scree plot.
    """
    # ── step 1: find how many PCA components explain enough variance ───────────
    print("fitting PCA on all embeddings to find the right number of components...")
    pca_probe = PCA()
    pca_probe.fit(all_embs)

    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    # searchsorted finds the first index where cumulative variance crosses the threshold
    k = int(np.searchsorted(cumvar, n_components_target)) + 1
    print(f"PCA: {k} components explain {n_components_target*100:.0f}% of variance (out of {all_embs.shape[1]} dims)")

    # ── save the scree plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, linewidth=1.5)
    ax.axhline(n_components_target, linestyle="--", color="red",  label=f"{n_components_target*100:.0f}% threshold")
    ax.axvline(k,                    linestyle="--", color="orange", label=f"chosen k={k}")
    ax.set_xlabel("number of PCA components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA scree plot — choosing k for t-SNE preprocessing")
    ax.legend()
    plt.tight_layout()
    scree_path = os.path.join(PLOTS_DIR, "pca_scree.png")
    plt.savefig(scree_path, dpi=150)
    plt.close()
    print(f"saved scree plot -> {scree_path}")

    # ── step 2: reduce to k dims ───────────────────────────────────────────────
    reduced = PCA(n_components=k).fit_transform(all_embs)   # (N, k)

    # ── step 3: t-SNE to 2D ───────────────────────────────────────────────────
    # perplexity = sqrt(N) is a principled default (roughly: neighborhood size)
    N = len(reduced)
    perplexity = int(np.sqrt(N))
    print(f"t-SNE: N={N}, perplexity={perplexity} (sqrt of N), output=2D")

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    coords_2d = tsne.fit_transform(reduced)   # (N, 2)

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
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── load embeddings ────────────────────────────────────────────────────────
    print("loading embeddings...")
    flickr_embs  = np.load(IMG_EMB_PATH)     # (1000, 512) — Flickr30K images
    gundam_base  = np.load(VAL_BASE_PATH)    # (N_val, 512) — Gundam, base CLIP
    gundam_ft    = np.load(VAL_FT_PATH)      # (N_val, 512) — Gundam, fine-tuned
    gundam_labels = np.load(VAL_LABELS_PATH) # (N_val,) — 0..5 series indices

    # shift Gundam labels by 1 so Flickr gets label 0 and Gundam series get 1–6
    gundam_labels_shifted = gundam_labels + 1

    N_flickr = len(flickr_embs)
    N_gundam = len(gundam_base)
    print(f"Flickr: {N_flickr} images, Gundam val: {N_gundam} images")

    # ── stack into one matrix for joint PCA+t-SNE ─────────────────────────────
    # Flickr label = 0, Gundam labels = 1..6
    all_labels = np.concatenate([
        np.zeros(N_flickr, dtype=int),
        gundam_labels_shifted,
    ])

    # run PCA+tSNE twice — once with base embeddings, once with fine-tuned
    print("\n--- base CLIP embeddings ---")
    all_base = np.vstack([flickr_embs, gundam_base])
    coords_base, k_base = run_pca_tsne(all_base)

    print("\n--- fine-tuned CLIP embeddings ---")
    all_ft = np.vstack([flickr_embs, gundam_ft])
    coords_ft, k_ft = run_pca_tsne(all_ft)

    # ── side-by-side before/after plot ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    make_scatter(axes[0], coords_base, all_labels, "Zero-shot CLIP")
    make_scatter(axes[1], coords_ft,   all_labels, "CLIP + LoRA (Gundam series)")

    # shared legend below both plots
    handles = [
        mpatches.Patch(color=COLORS[i], label=LABEL_NAMES[i])
        for i in sorted(LABEL_NAMES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Embedding space: Flickr30K (grey) + Gundam series (coloured)\n"
                 "LoRA fine-tuning teaches CLIP to cluster builds by series",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    tsne_path = os.path.join(PLOTS_DIR, "tsne_comparison.png")
    plt.savefig(tsne_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved t-SNE comparison -> {tsne_path}")
    print("done.")
