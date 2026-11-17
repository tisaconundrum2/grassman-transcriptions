# Grassmann Transcription — Agent Context

## What this project is
- Attention-free ASR (speech recognition) using Grassmann manifold geometry
- Based on arXiv:2512.19428 "Attention Is Not What You Need"
- Replaces self-attention with Plücker coordinate encoding on Gr(2,r)

## Architecture
- CausalGrassmannLayer: linear reduction → multi-scale pairing → Plücker encoding → gated fusion → FFN
- GrassmannConformerBlock: Macaron FFN sandwich + Grassmann mixer + ConvModule
- AblationLayer/AblationMixer: symmetric products (drop-in replacement, no Grassmann geometry)
- Conv2dSubsampling frontend (from Conformer paper)
- CTC loss for alignment-free training
- Two-sided (bidirectional) mode for encoder/CTC

## Key implementation details
- Uses (z_t, z_{t-delta}) NOT (z_t, z_{t+delta}) — causal pairing (past tokens)
- "Dhruv's variant" gating: h + (1-α)(g-h)
- Mean-first (anchored) Plücker computation — 3.5x faster
- Plücker antisymmetry distinguishes past/future in two-sided mode
- Window offsets: [1, 2, 4, 8, 12, 16]
- Grassmann: a_i*b_j - a_j*b_i (antisymmetric, Plücker coordinates)
- Ablation: a_i*b_j + a_j*b_i (symmetric, no manifold structure)
- Ablation flip=+1 for two-sided (vs -1 for Plücker), symmetric_norm_sq for weighting
- Matches Dhruv's grassmann-speech repo exactly

## Project structure
```
app.py                    # Unified CLI: train, cache, export, eval, transcribe, demo
recipes.yaml              # Dataset schemas + recipes as config (not hardcoded)
Dockerfile.grassman       # Grassmann training job (Plücker mixer)
Dockerfile.ablation       # Ablation training job (symmetric mixer)
Dockerfile.cache          # CPU mel-spectrogram caching job
src/
  models/
    grassmann_layer.py    # PluckerEncoder, GrassmannMixer, CausalGrassmannLayer
    ablation_layer.py     # SymmetricEncoder, AblationMixer, AblationLayer
    conformer_block.py    # GrassmannConformerBlock, AblationConformerBlock
    grassmann_model.py    # Full model — selects mixer via config.use_ablation
    config.py             # GrassmannConfig with use_ablation flag
  repositories/
    dataset_repository.py # HF Hub fetching, schema resolution, concurrent mel transforms
    cache_repository.py   # Disk cache + incremental shard uploads to HF Hub
  services/
    dataset_service.py    # Orchestration: load → preprocess → split → DataLoaders
    training_service.py   # Training loop, optimizer, scheduler
    export_service.py     # ONNX export (CTC and Seq2Seq)
    upload_service.py     # UploadThing direct upload
    tokenizer_service.py  # CharTokenizer
  controllers/
    model_controller.py   # Model build, predict, train orchestration
    pipeline_controller.py # End-to-end pipeline
```

## CLI flags
- `--ablation` — swaps Plücker for symmetric mixer
- `--run-name <name>` — prefixes saved artifacts (e.g. smol-grassmann-best_model.pt)
- `--max-samples <n>` — works with --from-cache to train on a subset
- `--from-cache <repo>` — download pre-cached mel shards from HF Dataset repo
- `--recipe <name>` — load dataset recipe from recipes.yaml
- `--workers <n>` — parallel mel workers (0=sequential, default for Docker)
- `--conformer` — use Conformer blocks (vs plain Grassmann layers)
- `--two-sided` — bidirectional mixing (for CTC/encoder mode)
- `--ctc` — CTC-only mode (vs Seq2Seq)

## Deployment — USE HF JOBS, NOT SPACES
**CRITICAL**: Always use `hf jobs run` for training, NOT HF Spaces. Spaces have ephemeral disk, secrets require restarts, and containers get killed by health checks.

### Launching a training job
```bash
hf jobs run \
  --flavor t4-small \
  --timeout 12h \
  --attempts 3 \
  --name smol-ablation \
  -s HF_TOKEN \
  --detach \
  registry.hf.space/tisaconundrum-grassmann-ablation:latest \
  -- python -u app.py train \
    --from-cache tisaconundrum/grassmann-mel-cache \
    --max-samples 28539 \
    --epochs 50 --batch-size 32 --layers 6 --d-model 256 --r-reduced 32 --lr 3e-4 \
    --ctc --conformer --two-sided --ablation \
    --run-name smol-ablation --save-dir checkpoints --decode-every 5
```

### Monitoring a job
```bash
hf jobs logs -f tisaconundrum/<job_id>     # Stream logs
hf jobs inspect tisaconundrum/<job_id>     # Check status
hf jobs list                               # List all jobs
```

### Docker images
Space repos auto-build Docker images at `registry.hf.space/<owner>-<space-name>:latest`:
- `registry.hf.space/tisaconundrum-grassmann-transcription:latest` — Grassmann mixer
- `registry.hf.space/tisaconundrum-grassmann-ablation:latest` — Ablation mixer

### HF CLI setup
```bash
pip install 'huggingface_hub>=1.8'
hf auth login --token <token>
```

## Data pipeline
1. **Cache job** (CPU): streams from HF datasets, computes mel spectrograms, uploads shards incrementally to `tisaconundrum/grassmann-mel-cache`
2. **Training job** (GPU): downloads shards from cache repo, `--max-samples` limits to subset
3. **Recipes**: defined in `recipes.yaml` — librispeech-100, librispeech-960, 1750h
4. **Incremental uploads**: shards upload every 20K samples, survives container restarts
5. **Checkpoint persistence**: best_model.pt + meta.json uploaded to HF repo every val_loss improvement, resumes on restart

## Checkpoint resume
- On startup, `_try_resume_checkpoint` checks `tisaconundrum/grassmann-mel-cache` for `checkpoints/{run_name}-best_model.pt`
- Downloads weights + metadata (epoch, val_loss), loads into model, resumes from next epoch
- Requires HF_TOKEN as env var for upload/download

## Training results
### Smol OG (original, 11M params)
- 50 epochs on train.clean.100 (100h, ~28K samples), T4 GPU, ~$4.40
- Epoch 45: train loss 0.071, val loss 0.736

### Smol Grassmann (re-run from cache)
- Epoch 29: val loss 0.062, checkpoint saved to HF repo

### Ablation vs Grassmann comparison
- Both track nearly identically through epoch 10 (val loss ~0.36 both)
- Ablation trains 2x faster per epoch (538s vs 1088s)
- KEY FINDING: mixer type barely matters at matched epochs
- Consistent with Dhruv's (grassmann-speech) finding that symmetric products tie with Plücker
- Implication: the Grassmann geometry isn't the active ingredient — fixed-distance multiplicative mixing is

## Scaling plan
- S (11M) ✅, M (50M), L (100M), XL (200M), Chonker (500M)
- Next: Medium 50M on full 960h LibriSpeech + beam search
- WER targets: <15% = publishable for attention-free arch

## Paper
- Private repo: tisaconundrum2/grassmann-asr-paper
- Dhruv (vakharwalad23) is collaborator
- If ablation ties: paper pivots from "Grassmann replaces attention" to "multiplicative mixing replaces attention"

## External references
- Dhruv's repo: grassmann-speech (Python + ESPnet integration)
- Paper: arXiv:2512.19428 "Attention Is Not What You Need"
- Browser demo: tisaconundrum2.github.io/grassman-transcriptions (ONNX + WASM)
