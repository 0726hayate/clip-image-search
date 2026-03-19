# CLIP Image-Text Retrieval on Flickr30K

Zero-shot image-text retrieval using OpenAI's CLIP (ViT-B/32) on the standard Flickr30K 1K test benchmark. Encodes 1000 images and 5000 captions into a shared 512-dim embedding space, then retrieves using cosine similarity.

The core idea: CLIP trains a vision encoder and a text encoder together so that matching image-caption pairs end up close in the embedding space. Once you have those embeddings you can do cross-modal retrieval — given a text query, find the most similar images, and vice versa.

## Results (zero-shot, no fine-tuning)

| Task           | R@1   | R@5   | R@10  |
|----------------|-------|-------|-------|
| Text → Image   | 58.8% | 83.5% | 90.0% |
| Image → Text   | 79.4% | 95.0% | 98.1% |

Evaluated on the Flickr30K 1K test set (1000 images × 5 captions = 5000 text queries). Matches published CLIP ViT-B/32 numbers from [Radford et al. 2021](https://arxiv.org/abs/2103.00020).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# step 1: download dataset and compute embeddings (~2 min on GPU)
python embed.py

# step 2: evaluate recall@k on the full benchmark
python retrieve.py

# step 3: interactive text search
python demo.py
```

## Example queries (demo.py)

```
query: a dog playing fetch on the beach
query: two children eating birthday cake
query: a man in a red jacket skiing down a mountain
```

## How it works

1. **embed.py** downloads the Flickr30K 1K test set from HuggingFace, extracts 1000 images and 5000 captions, and encodes them with CLIP. Embeddings are L2-normalized so cosine similarity equals a plain dot product.

2. **retrieve.py** builds the full similarity matrix (text × image or image × text) with a single matrix multiply, ranks by score, and computes Recall@K. For text→image: caption `t` belongs to image `t // 5`. For image→text: image `i` has 5 valid captions at flat indices `5i .. 5i+4`.

3. **demo.py** lets you type any English query and see the top-5 matching images with their ground truth captions.

## Model and dataset

- Model: `clip-ViT-B-32` via [sentence-transformers](https://www.sbert.net/) (wraps `openai/clip-vit-base-patch32`)
- Dataset: [`nlphuji/flickr_1k_test_image_text_retrieval`](https://huggingface.co/datasets/nlphuji/flickr_1k_test_image_text_retrieval) — the standard 1K test split of Flickr30K

---

## Domain-Specific Fine-Tuning: Gundam Series

To test whether CLIP can learn niche visual structure, I used LoRA to adapt the vision encoder for 6 Gundam mobile suit series. Each series has a strongly distinct visual design language (UC Unicorn's streamlined white/gold, AGE's blocky proportions, SEED's vibrant primary colours, etc.), making them a controlled probe for domain-specific embedding structure.

### Setup

- **Data**: 565 images scraped from the Gundam fandom wiki via MediaWiki API — 6 series (452 train / 113 val)
- **LoRA**: r=16, α=32, `target_modules=["q_proj","v_proj"]` (vision encoder attention only) — 983K trainable params, ≈0.6% of total
- **Loss**: Supervised Contrastive (Khosla et al. 2020) — all same-series images in a batch are positives for each other, cross-series are negatives. Richer signal than vanilla InfoNCE which has only one positive per anchor.
- **Training**: 10 epochs, batch size 32, lr=2e-4, temperature=0.07

```bash
# step 4: collect Gundam images from wiki (~5 min, requires internet)
python collect_gundam.py

# step 5: LoRA fine-tuning (~20 min on GPU, saves adapter to gundam_lora/)
python finetune.py

# step 6: PCA + t-SNE visualisation, saves plots/ directory
python visualize.py
```

### Results (Series Recall@K)

Given a mobile suit image, what fraction of queries have at least one same-series image in the top K?

| Model                | R@1   | R@5   | R@10  |
|----------------------|-------|-------|-------|
| Zero-shot CLIP       | 59.3% | 83.5% | 90.7% |
| CLIP + LoRA (Gundam) | 77.0% | 92.4% | 96.5% |

### Visualisation

PCA (k chosen by 95% explained variance) followed by t-SNE (perplexity = √N) on Flickr30K + Gundam val set:

![t-SNE comparison](plots/tsne_comparison.png)

After fine-tuning, the 6 Gundam series form tight, well-separated clusters. The Flickr images (grey) are largely unaffected — the adapter improves the niche domain without degrading general-purpose retrieval.

The PCA scree plot (saved to `plots/pca_scree.png`) shows that 95% of variance in CLIP's 512-dim space requires ~241 components — the embeddings are dense with no obvious low-rank shortcut, which is why the full PCA→t-SNE pipeline is necessary rather than going to t-SNE directly.
