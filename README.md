# CLIP Image-Text Retrieval + Domain Adaptation

Multi-modal image-text retrieval using CLIP-family models on the Flickr30K 1K test benchmark, with LoRA fine-tuning for domain-specific embedding structure.

The core idea: CLIP trains a vision encoder and a text encoder together so matching image-caption pairs end up close in the embedding space. Once you have those embeddings you can do cross-modal retrieval — given a text query, find the most similar images, and vice versa.

## Model Comparison (zero-shot, no fine-tuning)

Three controlled experiments against the CLIP ViT-B/32 baseline, varying vision and text encoder size independently to understand each component's contribution.

> Note: strict component isolation is impossible with jointly-trained models — "better vision" experiments also have slightly wider text encoders. Jina CLIP v2's text encoder scales 15× (38M→560M params, BERT-style bidirectional) while its vision encoder scales only 3.5× — making it a reasonable proxy for a text-dominant improvement.

### Text → Image (5000 queries, Flickr30K 1K test set)

| Experiment | Model | R@1 | R@5 | R@10 |
|------------|-------|-----|-----|------|
| baseline | CLIP ViT-B/32 | 58.8% | 83.5% | 90.0% |
| exp1: better vision | CLIP ViT-L/14 | 64.7% | 87.1% | 92.1% |
| exp2: better text | Jina CLIP v2 | 71.5% | 90.5% | 94.4% |
| exp3: both scaled | OpenCLIP ViT-H/14 | 77.7% | 94.2% | 96.6% |

### Image → Text (1000 queries)

| Experiment | Model | R@1 | R@5 | R@10 |
|------------|-------|-----|-----|------|
| baseline | CLIP ViT-B/32 | 79.4% | 95.0% | 98.1% |
| exp1: better vision | CLIP ViT-L/14 | 85.3% | 97.3% | 99.3% |
| exp2: better text | Jina CLIP v2 | 85.0% | 98.2% | 99.0% |
| exp3: both scaled | OpenCLIP ViT-H/14 | 90.6% | 99.2% | 99.7% |

**Key finding**: text→image retrieval benefits more from a better text encoder (Jina +12.7pp R@1) than a better vision encoder (L/14 +5.9pp). Image→text flips: L/14 and Jina tie at ~85%, suggesting the query-side encoder drives the gain.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# step 1: encode Flickr30K with all 4 models (~10 min on GPU)
python embed.py

# step 2: evaluate zero-shot retrieval
python retrieve.py

# step 3: interactive text search (uses CLIP ViT-B/32)
python demo.py
```

## Example queries (demo.py)

```
query: a dog playing fetch on the beach
query: two children eating birthday cake
query: a man in a red jacket skiing down a mountain
```

## How it works

1. **embed.py** downloads the Flickr30K 1K test set from HuggingFace, extracts 1000 images and 5000 captions, and encodes them with each of 4 models. Embeddings are L2-normalized so cosine similarity equals a plain dot product. Per-model files are saved as `img_embeddings_{tag}.npy` / `txt_embeddings_{tag}.npy`.

2. **retrieve.py** builds the full similarity matrix (text × image or image × text) with a single matrix multiply, ranks by score, and computes Recall@K for all 4 models. For text→image: caption `t` belongs to image `t // 5`. For image→text: image `i` has 5 valid captions at flat indices `5i .. 5i+4`.

3. **demo.py** lets you type any English query and see the top-5 matching images.

## Models

| Tag | Model | Vision | Text encoder | Emb dim |
|-----|-------|--------|-------------|---------|
| `b32` | `openai/clip-vit-base-patch32` | ViT-B/32, 12L | GPT-style 12L/512w (~38M) | 512 |
| `l14` | `openai/clip-vit-large-patch14` | ViT-L/14, 24L | GPT-style 12L/768w | 768 |
| `jina` | `jinaai/jina-clip-v2` | EVA02-L (307M) | XLM-RoBERTa-large (560M, bidirectional BERT-style) | 1024 |
| `h14` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | ViT-H/14, 32L | GPT-style 24L/1024w | 1024 |

---

## Domain-Specific Fine-Tuning: Gundam Series

To test whether CLIP can learn niche visual structure, LoRA was applied to each model's vision encoder for 6 Gundam mobile suit series. Each series has a strongly distinct visual design language, making them a controlled probe for domain-specific embedding structure.

### Setup

- **Data**: 565 images scraped from the Gundam fandom wiki via MediaWiki API — 6 series (452 train / 113 val)
- **LoRA**: r=16, α=32, `target_modules=["q_proj", "v_proj"]` (vision encoder attention), ~0.6–1% of parameters depending on model size
- **Loss**: Supervised Contrastive (Khosla et al. 2020) — all same-series images in a batch are positives for each other, cross-series are negatives
- **Training**: 10 epochs, batch size 32, lr=2e-4, temperature=0.07

```bash
# step 4: collect Gundam images from wiki (~5 min)
python collect_gundam.py

# step 5: LoRA fine-tuning — set MODEL_TAG at top of finetune.py
python finetune.py

# step 6: PCA + t-SNE visualisation for all models
python visualize.py
```

### Results (Series Recall@K)

Given a mobile suit image, what fraction of queries have at least one same-series image in the top K?

| Experiment | Model | Base R@1 | FT R@1 | Base R@10 | FT R@10 |
|------------|-------|----------|--------|-----------|---------|
| baseline | CLIP ViT-B/32 | 59.3% | 77.0% | 94.7% | 89.4% |
| exp1: better vision | CLIP ViT-L/14 | 67.3% | 85.8% | 95.6% | 94.7% |
| exp2: better text | Jina CLIP v2 | 56.6% | 82.3% | 93.8% | 94.7% |
| exp3: both scaled | OpenCLIP ViT-H/14 | 72.6% | 89.4% | 99.1% | 96.5% |

> Note: R@10 sometimes drops slightly after LoRA fine-tuning — SupCon loss tightens clusters and can push distant same-series images outside the top 10 while strongly improving top-1 precision. Jina shows the largest R@1 gain (+25.7pp) despite the lowest base score, suggesting its EVA02-L vision encoder adapts more easily than CLIP-trained encoders.

### Visualisation

PCA (k chosen by ≥95% explained variance) followed by t-SNE (perplexity = √N) on Flickr30K + Gundam val set. Each row is one model; left = zero-shot, right = LoRA fine-tuned:

![t-SNE comparison](plots/tsne_comparison.png)

Flickr30K images are colored by CLIP-assigned category (pastel, small dots); Gundam series are saturated large dots. After fine-tuning, the 6 Gundam series form tight, well-separated clusters while the Flickr category structure is largely preserved — the adapter improves the niche domain without degrading general-purpose retrieval.
