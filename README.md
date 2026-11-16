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

# Train from pre-cached mels (fastest — skips caching on GPU)
python app.py train --from-cache tisaconundrum/grassmann-mel-cache \
  --epochs 50 --batch-size 32 --ctc --conformer --two-sided

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

## GPU Training on HF Spaces

This repo is configured as a Docker Space. The workflow is:

1. **CPU Space** — `python app.py cache --recipe 1750h` streams audio, computes mels, uploads shards to a HF Dataset repo
2. **GPU Space** — `python app.py train --from-cache <repo>` downloads pre-cached mels and trains immediately (no expensive re-caching)

## Project Structure

```
src/
  interfaces/           # Abstract contracts (IModel, IDataset, etc.)
  models/               # Grassmann architecture (CausalGrassmannLayer, PluckerEncoder)
  services/             # Audio, datasets, training, ONNX export, WER eval, upload
  controllers/          # Fluent API orchestration
  views/                # Console output
app.py                  # Unified CLI (train, cache, export, eval, transcribe, demo)
inference_onnx.py       # Standalone ONNX inference (no PyTorch dependency)
demo/                   # Browser demo (GitHub Pages)
```

## License

MIT, see [LICENSE](LICENSE). Data and third-party code keep their own licenses.
