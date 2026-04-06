# CLIP Image-Text Retrieval + Adapter Swarm

Multi-modal image-text retrieval using CLIP-family models on the Flickr30K 1K test benchmark, plus a Pinterest-inspired **Adapter Swarm**: one base CLIP model paired with a library of lightweight LoRA adapters, each specializing for a different niche domain (Gundam mobile suits, Pokémon types, classical paintings).

The core idea: CLIP trains a vision encoder and a text encoder together so matching image-caption pairs end up close in the embedding space. Once you have those embeddings you can do cross-modal retrieval — given a text query, find the most similar images, and vice versa. The swarm pattern shows how to specialize that base for niche communities at production scale without serving N copies of the model.

---

## Adapter Swarm: niche specialization at production scale

Pinterest has hundreds of specialized communities — fashion, food, anime, art, woodworking — each with its own visual conventions that a generic CLIP doesn't fully capture. Fine-tuning a separate CLIP per niche is impractical: a 5 GB base model × N niches = N copies in your serving fleet. LoRA flips this:

- **One base model in memory** (OpenCLIP ViT-H/14, ~3 GB)
- **One ~10 MB adapter per niche** (Gundam, Pokémon, Paintings, ...)
- **Hot-swap the adapter at request time** (~hundreds of ms, not the seconds it takes to reload a full model)

Only the vision encoder gets LoRA'd — the text encoder is frozen across all adapters, so a query like *"fire dragon"* is encoded once with the base text encoder and matched against vision embeddings produced by each niche's adapter.

### Demo

```bash
# build per-niche image indexes (one-time, encodes each niche's images with its adapter)
python demo_swarm.py --build

# search a niche by text query
python demo_swarm.py --niche pokemon   --query "fiery red dragon"
python demo_swarm.py --niche paintings --query "moonlit sky"
python demo_swarm.py --niche gundam    --query "white red mobile suit"

# print storage / latency / quality benchmark tables
python swarm_analysis.py
```

### Three niches × one base

| Niche | Classes | Images | Source |
|---|---|---|---|
| Gundam | 6 series (UC, AGE, WFM, 00, IBO, G-Reco) | 541 | Gundam fandom wiki via MediaWiki API |
| Pokémon | 8 types (Fire/Water/Grass/Electric/Psychic/Dark/Dragon/Fairy) | 410 | PokéAPI official artwork |
| Paintings | 8 art movements (Renaissance → Pop Art) | 480 | huggan/wikiart on HuggingFace |

`swarm_analysis.py` outputs three tables (numbers measured on this machine, OpenCLIP H/14 base):

**Storage**
| Approach | Disk |
|---|---|
| 1 base h14 + 3 LoRA adapters (16 MB each) | **7.40 GB** |
| Alternative: 3 full FT'd h14 models | 22.06 GB |
| **Savings** | **66.5% / 3.0× storage efficiency** |

**Latency**
| Operation | Time |
|---|---|
| Cold-load base h14 from cache → GPU | 2889 ms |
| Hot adapter swap (base in memory, 3-niche avg) | **362 ms** |
| Encode 1 text query (base text encoder) | 692 ms |
| Search top-5 over 410 images | 0.08 ms |

Hot-swap is ~8× faster than reloading a full 3 GB model from disk. Combined with text encode + search, end-to-end query latency is ~1 sec from a cold base, dominated by text encoding — well under any per-request budget once the base is warm.

**Quality (Recall@K per niche)**
| Niche | Base R@1 | FT R@1 | Δ R@1 | Base R@10 | FT R@10 | Δ R@10 |
|---|---|---|---|---|---|---|
| Gundam | 67.9% | **90.8%** | +22.9pp | 97.2% | 97.2% | +0.0pp |
| Pokémon | 50.0% | 53.7% | +3.7pp | 90.2% | 87.8% | -2.4pp |
| Paintings | 79.2% | **94.8%** | +15.6pp | 95.8% | 96.9% | +1.0pp |

> Note: not all niches benefit equally from LoRA. Gundam series have a strongly conditioned visual design language (mobile suits in one series share silhouette conventions, paint schemes, joint articulation), so contrastive fine-tuning produces clean clusters. Pokémon **types** are visually noisier — a Fire-type can look like almost anything as long as it has flame elements — so the gain from LoRA is smaller. This is itself a useful signal: the swarm pattern works best for niches where the label correlates with visual structure.

---

## 4-model encoder ablation (zero-shot Flickr30K)

Three controlled experiments against the CLIP ViT-B/32 baseline, varying vision and text encoder size independently to understand each component's contribution.

> Note: strict component isolation is impossible with jointly-trained models — "better vision" experiments also have slightly wider text encoders. Jina CLIP v2's text encoder scales 15× (38M→560M params, BERT-style bidirectional) while its vision encoder scales only 3.5× — making it a reasonable proxy for a text-dominant improvement.

### Text → Image (5000 queries, Flickr30K 1K test set)

| Experiment | Model | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| baseline | CLIP ViT-B/32 | 58.8% | 83.5% | 90.0% |
| exp1: better vision | CLIP ViT-L/14 | 64.7% | 87.1% | 92.1% |
| exp2: better text | Jina CLIP v2 | 71.5% | 90.5% | 94.4% |
| exp3: both scaled | OpenCLIP ViT-H/14 | 77.7% | 94.2% | 96.6% |

### Image → Text (1000 queries)

| Experiment | Model | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| baseline | CLIP ViT-B/32 | 79.4% | 95.0% | 98.1% |
| exp1: better vision | CLIP ViT-L/14 | 85.3% | 97.3% | 99.3% |
| exp2: better text | Jina CLIP v2 | 85.0% | 98.2% | 99.0% |
| exp3: both scaled | OpenCLIP ViT-H/14 | 90.6% | 99.2% | 99.7% |

**Key finding**: text→image retrieval benefits more from a better text encoder (Jina +12.7pp R@1) than a better vision encoder (L/14 +5.9pp). Image→text flips: L/14 and Jina tie at ~85%, suggesting the query-side encoder drives the gain.

The 4-model comparison answers *which encoder family is best*. The swarm answers *how to deploy specialization at scale* — the two halves of the project are independent.

---

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

# step 3: interactive text search over Flickr (uses CLIP ViT-B/32)
python demo.py

# step 4: collect niche datasets
python collect_gundam.py        # Gundam series (wiki API)
python collect_pokemon.py       # Pokémon by type (PokéAPI)
python collect_paintings.py     # Art movements (huggan/wikiart streaming)

# step 5: LoRA fine-tune — args are <model_tag> <dataset_tag>
python finetune.py b32 gundam
python finetune.py l14 gundam
python finetune.py h14 gundam
python finetune.py jina gundam
python finetune.py h14 pokemon
python finetune.py h14 paintings

# step 6: encode Imagenette (held-out general dataset) with each model, base + LoRA
python embed_imagenette.py

# step 7: t-SNE visualizations (two figures, one per dataset, independent PCA+tSNE)
python visualize.py

# step 8: build swarm indexes + try queries
python demo_swarm.py --build
python demo_swarm.py --niche pokemon --query "fiery red dragon"

# step 9: storage / latency / quality benchmarks for the swarm
python swarm_analysis.py
```

---

## Models

| Tag | Model | Vision | Text encoder | Emb dim |
|---|---|---|---|---|
| `b32` | `openai/clip-vit-base-patch32` | ViT-B/32, 12L | GPT-style 12L/512w (~38M) | 512 |
| `l14` | `openai/clip-vit-large-patch14` | ViT-L/14, 24L | GPT-style 12L/768w | 768 |
| `jina` | `jinaai/jina-clip-v2` | EVA02-L (307M) | XLM-RoBERTa-large (560M, bidirectional BERT-style) | 1024 |
| `h14` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | ViT-H/14, 32L | GPT-style 24L/1024w | 1024 |

---

## How it works

1. **embed.py** downloads the Flickr30K 1K test set from HuggingFace, extracts 1000 images and 5000 captions, and encodes them with each of 4 models. Embeddings are L2-normalized so cosine similarity equals a plain dot product. Per-model files are saved as `img_embeddings_{tag}.npy` / `txt_embeddings_{tag}.npy`.

2. **retrieve.py** builds the full similarity matrix (text × image or image × text) with a single matrix multiply, ranks by score, and computes Recall@K for all 4 models. For text→image: caption `t` belongs to image `t // 5`. For image→text: image `i` has 5 valid captions at flat indices `5i .. 5i+4`.

3. **demo.py** lets you type any English query and see the top-5 matching Flickr images.

4. **finetune.py** is parameterized by `<model_tag> <dataset_tag>` and reads `<dataset_tag>/labels.json` for the class set. LoRA settings: r=16, α=32, `target_modules=["q_proj", "v_proj"]` for CLIP-style models (~0.4–1.0% of parameters); for Jina EVA02 the targets are `attn.proj`/`mlp.fc1`/`mlp.fc2` instead (EVA02's attention bypasses module-level LoRA on `q_proj`/`v_proj` — it accesses `.weight` directly, so adapters there are silently ignored). Loss: Supervised Contrastive (Khosla et al. 2020) — all same-class images in a batch are positives for each other, cross-class are negatives. Trained for 10 epochs at lr=2e-4 with τ=0.07.

5. **embed_imagenette.py** encodes 1000 Imagenette images (10 classes × 100) with each model — first base, then base + LoRA adapter loaded — to verify visually (in the t-SNE) that the Gundam fine-tune doesn't destroy general-purpose semantic clustering.

6. **demo_swarm.py** loads the H/14 base once, then for each niche encodes every image with that niche's adapter and saves the resulting index. At search time, the text encoder (frozen, base) encodes the query and a numpy `argpartition` returns the top-K matches from the relevant niche's pre-computed index.

7. **swarm_analysis.py** measures the actual sizes and latencies on this machine: base + 3 adapters vs 3 full FT models (storage), cold load vs hot adapter swap (latency), and per-niche Recall@K (quality).

---

## Domain-Specific Fine-Tuning Results

### Gundam Series Recall@K (4-model comparison)

Given a mobile suit image, what fraction of queries have at least one same-series image in the top K?

| Experiment | Model | Base R@1 | FT R@1 | Base R@10 | FT R@10 |
|---|---|---|---|---|---|
| baseline | CLIP ViT-B/32 | 55.0% | 69.7% | 94.5% | 92.7% |
| exp1: better vision | CLIP ViT-L/14 | 61.5% | 95.4% | 97.2% | 97.2% |
| exp2: better text | Jina CLIP v2 | 64.2% | 83.5% | 91.7% | 97.2% |
| exp3: both scaled | OpenCLIP ViT-H/14 | 67.9% | 90.8% | 97.2% | 97.2% |

> Note: R@10 sometimes drops slightly after LoRA fine-tuning — SupCon loss tightens clusters and can push distant same-series images outside the top 10 while strongly improving top-1 precision. L/14 shows the largest R@1 gain (+33.9pp); jina also shows a strong +5.5pp R@10 improvement, suggesting the EVA02-L vision encoder benefits noticeably from contrastive specialization.

### Visualization 1 — Gundam clustering (LoRA target domain)

PCA (k chosen by ≥95% explained variance) followed by t-SNE (perplexity = √N), run **independently per model and per dataset**. Each row is one model; left = base, right = LoRA fine-tuned:

![Gundam t-SNE](plots/tsne_gundam.png)

The 6 Gundam series form tighter, more separated clusters after LoRA fine-tuning across all models.

### Visualization 2 — Imagenette (held-out general dataset)

The same models encode 1000 Imagenette images (10 ImageNet classes with built-in labels — no prompt-engineering hack). Left = base, right = with the Gundam-trained LoRA loaded:

![Imagenette t-SNE](plots/tsne_imagenette.png)

The 10-class structure is preserved before vs after LoRA — the Gundam-domain adapter doesn't degrade general semantic clustering. (This is a sanity check for the swarm pattern: niche specialization shouldn't break the base for held-out queries.)
