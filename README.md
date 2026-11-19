---
title: Grassmann Transcription
emoji: 🔊
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# Grassmann Manifold Audio Transcription

**An attention-free speech recognition model using Grassmann flows and Plücker coordinates.**

Based on ["Attention Is Not What You Need"](https://arxiv.org/abs/2512.19428) — replaces self-attention with geometric operations on the Grassmann manifold Gr(2, r).

## Architecture

Instead of computing an L×L attention matrix, each Causal Grassmann layer:

1. **Linear reduction**: h ∈ ℝᵈ → z ∈ ℝʳ (r=32)
2. **Multi-scale pairing**: pairs (zₜ, zₜ₊Δ) across windows {1, 2, 4, 8, 12, 16}
3. **Plücker encoding**: computes C(r,2)=496 coordinates encoding 2D subspaces on Gr(2,r)
4. **Gated fusion**: α⊙h + (1-α)⊙g with learned gating
5. **Feed-forward**: standard FFN with GELU + residual + LayerNorm

CTC loss for alignment-free training on audio transcription.

## Quick Start

```bash
# Demo (synthetic data, verifies pipeline)
python app.py demo

# Train on LibriSpeech (streaming, no huge downloads)
python app.py train --dataset openslr/librispeech_asr \
  --split train.clean.100 --max-samples 500 --epochs 10

# Train from pre-cached mels (streams shards from HF Hub, no disk needed)
python app.py train --from-cache tisaconundrum/grassmann-mel-cache \
  --epochs 50 --batch-size 128 --ctc --conformer --two-sided

# Cache dataset on CPU, upload to HF Hub
python app.py cache --recipe 1750h --repo tisaconundrum/grassmann-mel-cache

# Export to ONNX
python app.py export --model checkpoints/bundle --output onnx_export/ --verify

# Evaluate WER on LibriSpeech test-clean
python app.py eval --model eval_model/best_model.pt --config eval_model/config.json

# Transcribe audio
python app.py transcribe --model checkpoints/bundle --audio speech.wav
```

## Browser Demo

Try it live: **[tisaconundrum2.github.io/grassman-transcriptions](https://tisaconundrum2.github.io/grassman-transcriptions/)**

Runs entirely in the browser via ONNX + WASM. 11M params, 120x real-time on 16min audio.

## GPU Training on HF Jobs

Training runs on [HF Jobs](https://huggingface.co/docs/hub/en/jobs) — per-second GPU billing, persistent secrets, auto-retry on failure. **Do not use HF Spaces for training** (ephemeral disk, health check kills, secrets require restarts).

Features: AMP (mixed precision) auto-enabled on GPU, shard streaming (local mount, Hub prefetch, or sequential), checkpoint resume with LR schedule restore.

### Setup

```bash
pip install 'huggingface_hub>=1.8'
hf auth login
```

All jobs use a single Docker image: `ghcr.io/tisaconundrum2/grassmann:latest`

### Step 1: Cache mel spectrograms (one-time, GPU recommended)

```bash
hf jobs run \
  --flavor t4-small \
  --timeout 6h \
  --name cache-1750h \
  -s HF_TOKEN \
  ghcr.io/tisaconundrum2/grassmann:latest \
  -- python -u app.py cache \
    --recipe 1750h \
    --cache-dir mel_cache \
    --repo tisaconundrum/grassmann-mel-cache \
    --samples-per-shard 5000
```

> Smaller shards (~1.8GB each at 5K samples) load faster, fit better in the mount cache, and give finer shuffle granularity.

### Step 2: Train (GPU)

**Smol Grassmann (100h, Plücker mixer):**
```bash
hf jobs run \
  --flavor t4-small \
  --timeout 12h \
  --attempts 3 \
  --name smol-grassmann \
  -s HF_TOKEN \
  --detach \
  ghcr.io/tisaconundrum2/grassmann:latest \
  -- python -u app.py train \
    --from-cache tisaconundrum/grassmann-mel-cache \
    --max-samples 28539 \
    --epochs 50 --batch-size 32 --layers 6 --d-model 256 --r-reduced 32 --lr 3e-4 \
    --ctc --conformer --two-sided \
    --run-name smol-grassmann --save-dir checkpoints --decode-every 5
```

**Smol Ablation (100h, symmetric mixer):**
```bash
hf jobs run \
  --flavor t4-small \
  --timeout 12h \
  --attempts 3 \
  --name smol-ablation \
  -s HF_TOKEN \
  --detach \
  ghcr.io/tisaconundrum2/grassmann:latest \
  -- python -u app.py train \
    --from-cache tisaconundrum/grassmann-mel-cache \
    --max-samples 28539 \
    --epochs 50 --batch-size 32 --layers 6 --d-model 256 --r-reduced 32 --lr 3e-4 \
    --ctc --conformer --two-sided --ablation \
    --run-name smol-ablation --save-dir checkpoints --decode-every 5
```

**Ablation 1750h (full cache, mounted volume, AMP):**
```bash
hf jobs run \
  --flavor t4-small \
  --timeout 48h \
  --attempts 3 \
  --name ablation-1750h \
  -s HF_TOKEN \
  -v hf://datasets/tisaconundrum/grassmann-mel-cache:/mnt/mel-cache:ro \
  --detach \
  ghcr.io/tisaconundrum2/grassmann:latest \
  -- python -u app.py train \
    --from-cache /mnt/mel-cache \
    --checkpoint-repo tisaconundrum/grassmann-mel-cache \
    --epochs 20 --batch-size 128 --layers 6 --d-model 256 --r-reduced 32 --lr 3e-4 \
    --ctc --conformer --two-sided --ablation \
    --run-name ablation-1750h --save-dir checkpoints --decode-every 5
```

> **Data loading modes:** `--from-cache` accepts a HF repo id or a local/mounted path. Use `--fetch-mode` to pick the strategy: `auto` (mount if path exists, else prefetch), `local`, `stream`, or `prefetch`. When using a mounted path, pass `--checkpoint-repo` for checkpoint upload/resume. Batch 128 is the max for T4 (15GB VRAM). AMP is auto-enabled on GPU.

### Step 3: Monitor

```bash
hf jobs logs -f tisaconundrum/<job_id>     # Stream logs
hf jobs inspect tisaconundrum/<job_id>     # Check status
hf jobs list                               # List all jobs
```

### Key flags

| Flag | Purpose |
|------|---------|
| `--from-cache <source>` | Pre-cached shards: HF repo id or local/mounted path |
| `--fetch-mode <mode>` | Shard fetcher: `auto`, `local`, `stream`, `prefetch` |
| `--checkpoint-repo <repo>` | HF repo for checkpoint upload/resume (auto when `--from-cache` is a repo id) |
| `--max-samples <n>` | Subset of cached data (28539 = train.clean.100) |
| `--ablation` | Swap Plücker for symmetric products |
| `--run-name <name>` | Prefix for saved artifacts (avoids collisions) |
| `--recipe <name>` | Dataset recipe from `recipes.yaml` |
| `--samples-per-shard <n>` | Samples per shard file (default 5000, cache subcommand) |
| `--workers <n>` | DataLoader num_workers (ignored for streaming) |
| `-s HF_TOKEN` | Pass HF token as secret (for checkpoint uploads) |
| `-v hf://datasets/<repo>:/path:ro` | Mount dataset as a local read-only volume |
| `--attempts 3` | Auto-retry on failure (checkpoints resume from Hub) |

### Checkpoint persistence

Best model weights + optimizer/scheduler state upload to `tisaconundrum/grassmann-mel-cache` under `checkpoints/{run_name}-best_model.pt` every time val_loss improves. On restart (or `--attempts` retry), training resumes from the last uploaded checkpoint with the correct LR schedule position.

### Docker image

One image for all jobs, built by GitHub Actions and pushed to GHCR:

```
ghcr.io/tisaconundrum2/grassmann:latest
```

The command passed to `hf jobs run` determines what it does (train, cache, eval, etc.).

### Cost

| Config | Hardware | Time | Cost |
|--------|----------|------|------|
| Smol (11M, 100h, 50 epochs, fp32) | T4 small | ~9h | ~$3.60 |
| Smol (11M, 100h, 50 epochs, AMP) | T4 small | ~4.2h | ~$1.70 |
| Ablation 1750h (11M, 20 epochs, AMP + mount) | T4 small | ~28h | ~$17 |
| Cache (1750h recipe, GPU) | T4 small | ~2h | ~$1.20 |

> AMP (mixed precision) gives ~1.8x speedup on T4. Mounted volumes avoid download overhead; smaller shards (5K samples) reduce load stalls.

## Project Structure

```
src/
  interfaces/           # Abstract contracts (IModel, IDataset, IShardRepository)
  models/               # Grassmann architecture (CausalGrassmannLayer, PluckerEncoder)
  repositories/         # Data access (shard fetching, HF Hub cache, dataset streaming)
  services/             # Audio, datasets, training, checkpoints, ONNX export, WER eval
  controllers/          # Fluent API orchestration (ModelController, PipelineController)
  views/                # Console output
app.py                  # Unified CLI (train, cache, export, eval, transcribe, demo)
inference_onnx.py       # Standalone ONNX inference (no PyTorch dependency)
demo/                   # Browser demo (GitHub Pages)
```

## License

MIT, see [LICENSE](LICENSE). Data and third-party code keep their own licenses.
