# CLIP Image-Text Retrieval on Flickr30K

Zero-shot image-text retrieval using OpenAI's CLIP (ViT-B/32) on the standard Flickr30K 1K test benchmark. Encodes 1000 images and 5000 captions into a shared 512-dim embedding space, then retrieves using cosine similarity.

The core idea: CLIP trains a vision encoder and a text encoder together so that matching image-caption pairs end up close in the embedding space. Once you have those embeddings you can do cross-modal retrieval — given a text query, find the most similar images, and vice versa.

## Results (zero-shot, no fine-tuning)

| Task           | R@1   | R@5   | R@10  |
|----------------|-------|-------|-------|
| Text → Image   | 65.6% | 87.0% | 92.0% |
| Image → Text   | 84.5% | 96.0% | 98.8% |

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
