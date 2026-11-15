---
title: Grassmann Transcription
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: t4-small
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
python app.py train-hf --dataset openslr/librispeech_asr \
  --split train.clean.100 --max-samples 500 --epochs 10

# Full GPU training
python train_gpu.py --epochs 50 --batch-size 32
```

## GPU Training on HF Spaces

This repo is configured as a Docker Space. Attach a T4 or L4 GPU and it will
automatically train on LibriSpeech clean-100.

## Project Structure

```
src/
  interfaces/           # Abstract contracts (IModel, IDataset, etc.)
  models/               # Grassmann architecture (CausalGrassmannLayer, PluckerEncoder)
  services/             # Audio processing, HF datasets, training loop
  controllers/          # Fluent API orchestration
  views/                # Console output
app.py                  # CLI entry point
train_gpu.py            # GPU training script (for Spaces)
```

## License

MIT, see [LICENSE](LICENSE). Data and third-party code keep their own licenses.
