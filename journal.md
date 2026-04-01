# Project Journal

## March 18

Starting this project to learn more about multi-modal embeddings. I've been reading about how Pinterest uses visual search to recommend pins — the core idea is that you embed images and text into the same space so you can do cross-modal retrieval. CLIP does exactly this, so seems like a good thing to implement from scratch (well, using the pretrained model).

Plan:
- Use the Flickr30K dataset which has 1000 images each with 5 human-written captions
- Encode everything with CLIP
- For evaluation use Recall@K which is the standard metric — given a caption, can you retrieve the right image in your top K results?

Going to try to keep this simple. No fancy reranking or fine-tuning, just the zero-shot baseline first.

## March 21

Ran into an annoying issue with the dataset loading. I was trying to use `load_dataset("nlphuji/flickr30k")` but it throws a RuntimeError about deprecated loading scripts in the newer version of the datasets library. Spent like an hour on this before I found that they have a separate 1K test repo (`nlphuji/flickr_1k_test_image_text_retrieval`) which you can just download directly with `hf_hub_download`. That works fine and is actually cleaner since it gives you a CSV + zip directly.

The CSV has a `raw` column that stores the 5 captions as a JSON string inside a string, so you need `ast.literal_eval` to parse it. A bit weird but whatever.

Image embeddings are done (1000 × 512 float32). Text embeddings running now. Going to add the retrieval logic tomorrow.

## March 23

Got the evaluation working. The results are honestly better than I expected for zero-shot:

```
text -> image:  R@1=58.8%  R@5=83.5%  R@10=90.0%
image -> text:  R@1=79.4%  R@5=95.0%  R@10=98.1%
```

R@10 of ~90% for text-to-image means that if you type a caption, there's a 90% chance the right image is in the top 10 results out of 1000. That's pretty impressive for a model that was never fine-tuned on this dataset.

The index math for the evaluation was a bit tricky — captions are stored flat (5000 total) so caption `t` belongs to image `t // 5`. Had to be careful about this mapping. The image-to-text direction is slightly different because each image has 5 valid captions, so you get a hit if any of them appear in the top K.

Tomorrow I'll add the interactive demo and clean up the README.

## March 24

Finished the Flickr part. Added the interactive demo (demo.py) — it's fast enough that there's no noticeable lag even without FAISS or any approximate search. For 1000 images the brute-force dot product is plenty fast.

The README now has the full results table and explains the index math behind the evaluation. Next I want to try fine-tuning on a niche domain to see if CLIP can learn domain-specific structure. Thinking about using Gundam series — each series (Unicorn, SEED, 00, etc.) has such a distinct visual aesthetic that it should be a good test.

## March 25

Found images on the Gundam fandom wiki organized by series. Using renders/official art for now since I'm still waiting on a Reddit API application to get real Gunpla build photos. The wiki images are lower quality as training data (official renders vs actual photographs of models) but good enough to test whether the approach works.

Collected 6 series: UC Unicorn, AGE, SEED, Gundam 00, IBO, and G-Reco. Each series has a pretty distinct visual design language so this should be a good clustering test.

## March 28

LoRA fine-tuning is working. Used SupCon loss (same as my research project but for images instead of code) — all same-series suits in a batch are positives for each other, cross-series are negatives. LoRA only touches ~1% of CLIP's parameters (q_proj and v_proj in the vision encoder attention layers) but the series retrieval noticeably improves.

Will write up the numbers once the visualization is done.

## April 1

t-SNE visualization is done and honestly it's the most satisfying part of this project. The before/after comparison shows the 6 Gundam series forming distinct clusters after fine-tuning, while the Flickr images barely move — meaning we improved the niche domain without hurting the general-purpose embeddings.

Used PCA first to find the right number of components systematically (explained variance ≥ 95%), then t-SNE with perplexity = sqrt(N). Added 7 Flickr categories (people, animals, sports, nature, food, vehicles, architecture) assigned by CLIP zero-shot similarity — the visualization now shows both Gundam clustering and Flickr semantic structure in the same plot. Updated the README with the new results section and embedded both plots.

## April 8

Realized the B/32 zero-shot results (R@1=58.8%) are a bit underwhelming on their own for a resume project. Started a 3-experiment comparison to understand how each encoder component contributes:

- Exp 1 (better vision): fix text encoder, scale vision → CLIP ViT-L/14 (12L→24L vision)
- Exp 2 (better text): fix vision, scale text → Jina CLIP v2 (EVA02-L vision + 560M XLM-RoBERTa text, 15× bigger text encoder than B/32's ~38M)
- Exp 3 (both scaled): OpenCLIP ViT-H/14 trained on LAION-2B-en

The Jina choice is deliberate — XLM-RoBERTa is bidirectional BERT-style vs CLIP's causal GPT-style text encoder, so exp 2 is also an architecture comparison, not just scale. True component isolation isn't possible with jointly-trained models so I added a caveat in the README.

Running embeddings for all 3 new models and will add LoRA fine-tuning for each to get a full Gundam comparison.

## April 12

Comparison done. Full results:

Flickr30K zero-shot (text→image R@1):
- B/32 baseline: 58.8%
- L/14 (exp1 better vision): 64.7% (+5.9pp)
- Jina CLIP v2 (exp2 better text): 71.5% (+12.7pp)  
- OpenCLIP H/14 (exp3 both): 77.7% (+18.9pp)

The text encoder story is the clearest: Jina's bidirectional XLM-RoBERTa (560M params) outperforms L/14's GPT-style text encoder by 6.8pp on text→image retrieval, even though L/14's vision encoder is substantially bigger. Makes sense — the query in text→image retrieval IS the text, so better text encoding directly improves the query representation.

Gundam LoRA fine-tuning R@1 gains:
- B/32: 59.3% → 77.0%
- L/14: 67.3% → 85.8%
- Jina: 56.6% → 82.3% (largest gain, +25.7pp)
- H/14: 72.6% → 89.4% (highest absolute)

Surprising: Jina shows the biggest R@1 improvement from LoRA despite having the lowest base score. The EVA02-L encoder wasn't trained with the same visual-semantic CLIP objective, so it has more room to reshape its visual embedding space with a small adapter.

Updated README with full comparison tables and regenerated the 4-row t-SNE plot with Flickr category colors.

## April 17

Swapping SEED for Witch from Mercury — wanted a more recent series in the Gundam set rather than the long-running franchise pages. Updated `collect_gundam.py` (one-line dict change) and the SERIES_TO_IDX map. Re-running fine-tuning on all 4 models since the val composition changes.

Did a small refactor too: `finetune.py` now takes `<model_tag> <dataset_tag>` as positional args and computes the class index map dynamically from `<dataset>/labels.json` — no more hardcoded SERIES_TO_IDX. This is groundwork for what I'm planning next (training adapters on more niches than just Gundam).

b32 done quickly. l14 + h14 going overnight on different GPUs. jina tomorrow because it's the slow one (512×512 input + EVA02-L = large activations during backprop, runs at batch_size 8 instead of 32).

WFM page only had 51 valid images (vs ~100 for the older series), so the WFM cluster will be the smallest one in the t-SNE — but contrastive learning works fine with that count.

## April 21

Looked at the t-SNE again and the Flickr categories were a mess — `sports` had eaten 46% of the images because the CLIP zero-shot prompt I'd written ("outdoor recreational activity") was way too broad. Flickr30K is heavily people-doing-things, so almost everything matched. Dropped the prompt-based categorization entirely.

Replaced with **Imagenette** — 10 ImageNet classes with built-in labels (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute). No prompt hack required. Used `johnowhitaker/imagenette2-320` on HuggingFace because the original `frgfm/imagenette` is now broken (uses a deprecated dataset script).

Also split the visualization into two figures — `tsne_gundam.png` and `tsne_imagenette.png` — each with its own PCA + tSNE. The old combined plot had the Gundam clusters dominating the variance and squashing the Flickr structure into a less-discriminated blob. Independent runs fix that.

Imagenette gets encoded with both base AND the Gundam LoRA loaded on top. The point is to verify visually that the niche fine-tune doesn't destroy general semantic clustering on a held-out dataset — the 10 classes should still form distinct color groups in both columns.

