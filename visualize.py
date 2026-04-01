"""Two independent t-SNE figures: one for Gundam, one for Imagenette.

Each figure is a 4×2 grid (4 models × base/ft). PCA + t-SNE is run
independently per model AND per dataset — the Gundam plot's t-SNE never
sees Imagenette data and vice versa, so they don't fight for variance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")

MODELS = {
    "b32":  "CLIP ViT-B/32\n(baseline)",
    "l14":  "CLIP ViT-L/14\n(exp1: better vision)",
    "jina": "Jina CLIP v2\n(exp2: better text)",
    "h14":  "OpenCLIP ViT-H/14\n(exp3: both scaled)",
}

# ── Gundam (6 series) ─────────────────────────────────────────────────────
# Class indices come from sorted set of label values in gundam/labels.json:
#   ["00", "age", "grg", "ibo", "uc", "wfm"]  →  0..5
GUNDAM_LABELS = {
    0: "Gundam 00",
    1: "AGE",
    2: "G-Reco",
    3: "IBO",
    4: "UC Unicorn",
    5: "Witch from Mercury",
}
GUNDAM_COLORS = {
    0: "gold",         # 00
    1: "darkorange",   # AGE
    2: "mediumpurple", # G-Reco
    3: "seagreen",     # IBO
    4: "steelblue",    # UC Unicorn
    5: "tomato",       # WFM
}
GUNDAM_SIZES  = {i: 28 for i in range(6)}
GUNDAM_ALPHAS = {i: 0.85 for i in range(6)}

# ── Imagenette (10 ImageNet classes, real labels — no prompt hack) ────────
IMAGENETTE_LABELS = {
    0: "tench", 1: "English springer", 2: "cassette player", 3: "chain saw",
    4: "church", 5: "French horn", 6: "garbage truck", 7: "gas pump",
    8: "golf ball", 9: "parachute",
}
IMAGENETTE_COLORS = {i: c for i, c in enumerate(plt.cm.tab10.colors)}
IMAGENETTE_SIZES  = {i: 14 for i in range(10)}
IMAGENETTE_ALPHAS = {i: 0.7 for i in range(10)}


def run_pca_tsne(all_embs, n_components_target=0.95, tag=""):
    """PCA (k by explained variance threshold) → t-SNE to 2D."""
    print(f"  [{tag}] PCA on {all_embs.shape}...")
    pca_probe = PCA().fit(all_embs)
    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, n_components_target)) + 1
    print(f"  [{tag}] k={k} explains {n_components_target*100:.0f}% variance")

    reduced = PCA(n_components=k).fit_transform(all_embs)
    N = len(reduced)
    perplexity = max(5, int(np.sqrt(N)))
    print(f"  [{tag}] t-SNE: N={N}, perplexity={perplexity}")
    coords_2d = TSNE(n_components=2, perplexity=perplexity,
                     max_iter=1000, random_state=42).fit_transform(reduced)
    return coords_2d


def make_scatter(ax, coords, labels, label_names, colors, sizes, alphas, title):
    for lid in sorted(set(labels.tolist())):
        mask = labels == lid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colors[lid], s=sizes[lid], alpha=alphas[lid],
                   label=label_names[lid], edgecolors="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def make_plot(dataset_tag, label_names, colors, sizes, alphas,
              labels_path, out_path, suptitle):
    """Generate a 4×2 t-SNE figure for one dataset, all available models."""
    available = []
    for tag in MODELS:
        base_p = os.path.join(SCRIPT_DIR, f"{dataset_tag}_base_{tag}.npy")
        ft_p   = os.path.join(SCRIPT_DIR, f"{dataset_tag}_ft_{tag}.npy")
        if os.path.exists(base_p) and os.path.exists(ft_p):
            available.append(tag)
        else:
            missing = [os.path.basename(p) for p in [base_p, ft_p] if not os.path.exists(p)]
            print(f"[{dataset_tag}/{tag}] missing: {missing}, skipping")

    if not available:
        print(f"no models with complete {dataset_tag} embeddings, skipping plot")
        return

    labels = np.load(os.path.join(SCRIPT_DIR, labels_path))

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, tag in enumerate(available):
        print(f"\n[{dataset_tag}/{tag}] {MODELS[tag].strip()}")
        base_e = np.load(os.path.join(SCRIPT_DIR, f"{dataset_tag}_base_{tag}.npy"))
        ft_e   = np.load(os.path.join(SCRIPT_DIR, f"{dataset_tag}_ft_{tag}.npy"))

        coords_base = run_pca_tsne(base_e, tag=f"{dataset_tag}_{tag}_base")
        coords_ft   = run_pca_tsne(ft_e,   tag=f"{dataset_tag}_{tag}_ft")

        model_label = MODELS[tag].replace("\n", " ")
        left_title  = f"{model_label} — base"
        right_title = f"{model_label} — LoRA fine-tuned"
        make_scatter(axes[row, 0], coords_base, labels, label_names,
                     colors, sizes, alphas, left_title)
        make_scatter(axes[row, 1], coords_ft,   labels, label_names,
                     colors, sizes, alphas, right_title)

    handles = [mpatches.Patch(color=colors[i], label=label_names[i])
               for i in sorted(label_names)]
    legend_cols = min(len(label_names), 6)
    fig.legend(handles=handles, loc="lower center", ncol=legend_cols,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(suptitle, fontsize=11, y=1.005)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1) Gundam clusters — primary domain (LoRA was trained for this)
    make_plot(
        dataset_tag   = "gundam_val",
        label_names   = GUNDAM_LABELS,
        colors        = GUNDAM_COLORS,
        sizes         = GUNDAM_SIZES,
        alphas        = GUNDAM_ALPHAS,
        labels_path   = "gundam_val_labels.npy",
        out_path      = os.path.join(PLOTS_DIR, "tsne_gundam.png"),
        suptitle      = ("Gundam series — base vs LoRA fine-tuned (PCA→t-SNE per model, independent)\n"
                         "6 series: Gundam 00 / AGE / G-Reco / IBO / UC Unicorn / Witch from Mercury"),
    )

    # 2) Imagenette — held-out general dataset (LoRA trained on Gundam, not Imagenette)
    make_plot(
        dataset_tag   = "imagenette",
        label_names   = IMAGENETTE_LABELS,
        colors        = IMAGENETTE_COLORS,
        sizes         = IMAGENETTE_SIZES,
        alphas        = IMAGENETTE_ALPHAS,
        labels_path   = "imagenette_labels.npy",
        out_path      = os.path.join(PLOTS_DIR, "tsne_imagenette.png"),
        suptitle      = ("Imagenette — base vs after Gundam-LoRA loaded (PCA→t-SNE per model, independent)\n"
                         "10 ImageNet classes encoded with both base and LoRA-loaded models — "
                         "structure should be preserved (the adapter is a niche specialization, not a replacement)"),
    )

    print("\ndone.")
