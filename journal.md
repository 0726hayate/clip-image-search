# Project Journal

## April 15

Starting this project to learn more about multi-modal embeddings. I've been reading about how Pinterest uses visual search to recommend pins — the core idea is that you embed images and text into the same space so you can do cross-modal retrieval. CLIP does exactly this, so seems like a good thing to implement from scratch (well, using the pretrained model).

Plan:
- Use the Flickr30K dataset which has 1000 images each with 5 human-written captions
- Encode everything with CLIP
- For evaluation use Recall@K which is the standard metric — given a caption, can you retrieve the right image in your top K results?

Going to try to keep this simple. No fancy reranking or fine-tuning, just the zero-shot baseline first.
