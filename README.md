# CLIP Image-Text Retrieval on Flickr30K

Playing around with CLIP for image-text retrieval. The idea is to embed images and captions into the same vector space, then use cosine similarity to find matches.

## What I'm building

- Download the Flickr30K 1K test set (1000 images, 5 captions each)
- Encode everything with CLIP (ViT-B/32)
- Search: given a text query, find the most relevant images
- Evaluate how well it does with standard Recall@K metrics

## Setup

```bash
pip install -r requirements.txt
```

## Usage

More to come once I finish the scripts. Plan is:

```bash
# encode images + captions
python embed.py

# run evaluation
python retrieve.py

# interactive search demo
python demo.py
```

## Dataset

Using the Flickr30K 1K test split from HuggingFace: `nlphuji/flickr_1k_test_image_text_retrieval`

## Model

CLIP ViT-B/32 via sentence-transformers
