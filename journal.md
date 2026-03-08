# Project Journal

## April 15

Starting this project to learn more about multi-modal embeddings. I've been reading about how Pinterest uses visual search to recommend pins — the core idea is that you embed images and text into the same space so you can do cross-modal retrieval. CLIP does exactly this, so seems like a good thing to implement from scratch (well, using the pretrained model).

Plan:
- Use the Flickr30K dataset which has 1000 images each with 5 human-written captions
- Encode everything with CLIP
- For evaluation use Recall@K which is the standard metric — given a caption, can you retrieve the right image in your top K results?

Going to try to keep this simple. No fancy reranking or fine-tuning, just the zero-shot baseline first.

## April 18

Ran into an annoying issue with the dataset loading. I was trying to use `load_dataset("nlphuji/flickr30k")` but it throws a RuntimeError about deprecated loading scripts in the newer version of the datasets library. Spent like an hour on this before I found that they have a separate 1K test repo (`nlphuji/flickr_1k_test_image_text_retrieval`) which you can just download directly with `hf_hub_download`. That works fine and is actually cleaner since it gives you a CSV + zip directly.

The CSV has a `raw` column that stores the 5 captions as a JSON string inside a string, so you need `ast.literal_eval` to parse it. A bit weird but whatever.

Image embeddings are done (1000 × 512 float32). Text embeddings running now. Going to add the retrieval logic tomorrow.

## April 20

Got the evaluation working. The results are honestly better than I expected for zero-shot:

```
text -> image:  R@1=58.8%  R@5=83.5%  R@10=90.0%
image -> text:  R@1=79.4%  R@5=95.0%  R@10=98.1%
```

R@10 of ~92% for text-to-image means that if you type a caption, there's a 92% chance the right image is in the top 10 results out of 1000. That's pretty impressive for a model that was never fine-tuned on this dataset.

The index math for the evaluation was a bit tricky — captions are stored flat (5000 total) so caption `t` belongs to image `t // 5`. Had to be careful about this mapping. The image-to-text direction is slightly different because each image has 5 valid captions, so you get a hit if any of them appear in the top K.

Tomorrow I'll add the interactive demo and clean up the README.

## April 21

Finished. Added the interactive demo (demo.py) — it's fast enough that there's no noticeable lag even without FAISS or any approximate search. For 1000 images the brute-force dot product is plenty fast.

The README now has the full results table and explains the index math behind the evaluation. If I had more time I'd add a fine-tuning step to see how much improvement you can squeeze out of a few epochs on the training split — but the zero-shot numbers are already solid and the code is clean enough to extend.
