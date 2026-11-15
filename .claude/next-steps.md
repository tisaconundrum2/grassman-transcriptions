# Next Steps Plan

## Completed Run (Smol - 11M)
- **Status**: COMPLETE — 50 epochs on train.clean.100 (~28K samples, 100 hours)
- **Config**: 6L Conformer-CTC, d=256, r=32, batch 32, lr 3e-4, 50 epochs
- **Hardware**: T4-small ($0.40/hr), GPU using only 0.2/0.3 GB of 16 GB
- **Cost**: ~$4.40
- **Artifacts**: Auto-uploads ONNX + model bundle to HF repo, then auto-pauses
- **Results at epoch 45**:
  - Train loss: 0.071, Val loss: 0.736 (10× gap = significant overfitting on 100h)
  - Common words solid, proper nouns still weak ("launcelot" → "lancolot")
  - Phonetic substitutions remain ("custle", "wagte", "whistmonst her")
  - Sample [2] nearly character-perfect on second half of utterance
- **Key insight**: Model has unused capacity (memorized 100h). More data would force generalizable features instead of memorization.

---

## Iteration 1: Evaluate Smol Results
- [ ] Compute WER on LibriSpeech test-clean using saved Smol checkpoint
- [ ] Measure greedy WER, then beam search WER (retroactive from checkpoint)
- [ ] Write up results for LinkedIn post

## Iteration 2: Medium + 960h + Beam Search (The Real Run)
Skip incremental steps — scale model, data, and decoding simultaneously.

| | Smol (completed) | Medium (next) |
|--|--|--|
| Params | 11M | ~50M |
| Data | 100h | 960h (full LibriSpeech) |
| Decoding | Greedy | Beam search + LM |
| d_model | 256 | 512 |
| Layers | 6 | 12 |
| r_reduced | 32 | 64 |
| Epochs | 50 | 50 (may need fewer — more data = less overfitting) |
| Time/Epoch | ~13 min | ~2.5 hrs (55 min × ~3× more data) |
| Total | ~11 hrs | ~125 hrs |
| Cost | ~$4 (T4) | ~$50 (T4) |

### Pre-launch TODO (3 pieces of work)

#### 1. Multi-split data loading
Concatenate all three LibriSpeech training splits:
```
train.clean.100  →  28K samples  (100h)
train.clean.360  → 104K samples  (360h)
train.other.500  → 148K samples  (500h)
─────────────────────────────────────────
Total:             280K samples  (960h)
```
- [ ] Update `train_gpu.py` to accept multiple splits
- [ ] Stream and concatenate all three splits
- [ ] DiskCachedDataset already exists; need HDF5 single-file cache for 280K samples
- [ ] Length-sorted bucketing to eliminate padding waste

#### 2. Beam search decoder
- [ ] Add `beam_decode()` alongside current greedy `predict()`
- [ ] Use `pyctcdecode` (wraps KenLM, does CTC beam search out of the box)
- [ ] Train or download a character n-gram LM for English
- [ ] Beam search is decode-time only — no retraining needed, applies to Smol too

#### 3. WER computation in training loop
- [ ] Add `jiwer` for proper WER/CER metrics during validation
- [ ] Log WER alongside loss each epoch

## Iteration 4: Multi-Dataset Training
Add normalized external datasets for more acoustic diversity:

| Dataset | Hours | Normalization |
|---------|-------|---------------|
| LibriSpeech 960h | 960 | None needed |
| Common Voice 17 (en) | ~2,800 | lowercase + strip punct |
| VoxPopuli (en) | 543 | lowercase + strip punct |
| GigaSpeech (S) | 250 | lowercase + strip punct |
| SPGISpeech | 5,000 | lowercase + strip punct |

Normalization function:
```python
def normalize_transcript(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
```

---

## Model Scaling Ladder

| Tier | Name | Params | d_model | Layers | r | GPU Est. | Time/Epoch | 50 Epochs | Cost |
|------|------|--------|---------|--------|---|----------|------------|-----------|------|
| S | Smol | 11M | 256 | 6 | 32 | 0.3 GB | ~13 min | ~11 hrs | ~$4 (T4) |
| M | Medium | 50M | 512 | 12 | 64 | ~2 GB | ~55 min | ~46 hrs | ~$18 (T4) |
| L | Large | 100M | 768 | 15 | 96 | ~5 GB | ~2 hrs | ~100 hrs | ~$40 (T4) |
| XL | X-Large | 200M | 1024 | 18 | 128 | ~10 GB | ~4 hrs | ~200 hrs | ~$80 (T4) |
| C | Chonker | 500M | 1536 | 24 | 192 | ~30 GB | ~1.5 hrs | ~75 hrs | ~$190 (A100) |

---

## Architecture Improvements (After Baseline)

### High Impact
- [ ] **BPE Tokenizer** — fewer output steps = better CTC alignment (already implemented, not enabled)
- [ ] **SpecAugment** — data augmentation (time/freq masking) for free regularization
- [ ] **Length-sorted bucketing** — eliminate padding waste, allow larger effective batch sizes
- [ ] **Mixed precision (FP16/BF16)** — halves GPU memory, enables larger models
- [ ] **Gradient accumulation** — simulate larger batch sizes without more GPU memory

### Medium Impact
- [ ] **Beam search + LM** — external character/word LM at decode time
- [ ] **BPE tokenizer** with 1000-5000 vocab for CTC
- [ ] **Warm-start from Smol** — initialize Medium's first 6 layers from Smol weights

### Research Questions
- [ ] Does two-sided vs causal actually matter for CTC? (Dhruv's question)
- [ ] Does the Grassmann geometry help, or is it just the multiplicative mixing? (Dhruv's symmetric control)
- [ ] How does WER scale with model size? (Smol → Medium → Large)
- [ ] Can we match a same-size Transformer Conformer? (the real benchmark)

---

## Infrastructure Improvements

### Already Done
- [x] Health server on port 7860 (prevents HF Spaces timeout kill)
- [x] Artifact upload to HF repo (ONNX + model bundle)
- [x] Auto-pause after training (stops billing)
- [x] GPU memory logging per epoch
- [x] `empty_cache()` between epochs
- [x] HF_TOKEN as Space secret
- [x] `git push hf main` workflow (new write token)
- [x] DiskCachedDataset (for future large-data runs)

### TODO
- [ ] **HDF5 single-file cache** — replace 28K .pt files with one .h5 for large datasets
- [ ] **Checkpointing to HF repo** — save best model mid-training (not just at end)
- [ ] **WER computation** — add jiwer/evaluate for proper WER/CER metrics in training loop
- [ ] **TensorBoard / W&B logging** — loss curves, LR schedule, sample transcriptions
- [ ] **Multi-split streaming** — load train.clean.100 + train.clean.360 + train.other.500

---

## WER Targets

| WER | What it means |
|-----|---------------|
| >60% | Model outputting mostly noise |
| 30-50% | Learned phonetic patterns, garbled words |
| 15-30% | Genuinely impressive for 11M params + no LM |
| <15% | Publishable result for attention-free architecture |
| 11.09% | Nemotron 3.5 ASR (3.5B params, massive data) |
| 6.11% | Qwen 1.7B (1.7B params, massive data) |

---

## Key Lessons Learned

1. **HF Spaces kills Docker containers that don't serve HTTP on port 7860 after ~30 min** — this was NOT an OOM, it was a health check timeout
2. **Each LibriSpeech mel spectrogram costs ~1 MB in RAM** — limits in-memory datasets to ~5-7K on T4's 15 GB
3. **Padding tax is real** — one 30s clip in a batch of 8 inflates memory 5x vs median-length clips
4. **GPU was barely used** (0.2/0.3 GB) — model is tiny, all the constraint was system RAM + health check
5. **Disk caching individual .pt files creates OS page cache pressure** — Dhruv's approach (single tensor/HDF5) is better
6. **`upload_folder` API and `git push hf` both need write tokens** — tokens can expire; use Space secrets
7. **Cosine LR with warmup** — 3 warmup epochs then cosine decay works well for CTC
