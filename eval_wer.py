#!/usr/bin/env python3
"""
WER Evaluation on LibriSpeech test-clean.

Downloads test-clean split, runs greedy CTC decode on every sample,
computes WER/CER using jiwer.

Usage:
    python eval_wer.py --model eval_model/best_model.pt --config eval_model/config.json
    python eval_wer.py --model eval_model/best_model.pt --config eval_model/config.json --max-samples 100
"""

import argparse
import json
import time

import torch
import torchaudio
from jiwer import wer, cer

from src.models.config import GrassmannConfig
from src.models.grassmann_model import GrassmannTranscriptionModel
from src.services.tokenizer_service import CharTokenizer


def load_model(model_path: str, config_path: str) -> tuple:
    """Load model from checkpoint + config."""
    with open(config_path) as f:
        config_dict = json.load(f)

    config = GrassmannConfig(**{
        k: v for k, v in config_dict.items()
        if k in GrassmannConfig.__dataclass_fields__
    })

    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    model = GrassmannTranscriptionModel(config)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, tokenizer


_mel_transform = None

def make_mel(waveform: torch.Tensor, sample_rate: int, config: GrassmannConfig) -> torch.Tensor:
    """Convert waveform to mel spectrogram (matches training pipeline exactly)."""
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
        )

    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel = _mel_transform(waveform)
    mel = torch.log(mel.clamp(min=1e-9))
    mel = mel.squeeze(0)  # (n_mels, time)

    # Per-utterance normalization (must match training!)
    mean = mel.mean()
    std = mel.std().clamp(min=1e-9)
    mel = (mel - mean) / std

    return mel.unsqueeze(0)  # (1, n_mels, time) for batch dim


@torch.no_grad()
def greedy_decode(model, mel: torch.Tensor, tokenizer: CharTokenizer) -> str:
    """Run greedy CTC decode on a single mel spectrogram."""
    log_probs = model(mel)  # (T, 1, vocab)
    preds = log_probs.argmax(dim=-1).squeeze(1)  # (T,)
    return tokenizer.decode(preds)


def main():
    parser = argparse.ArgumentParser(description="WER Evaluation")
    parser.add_argument("--model", default="eval_model/best_model.pt")
    parser.add_argument("--config", default="eval_model/config.json")
    parser.add_argument("--split", default="test.clean")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--show-errors", type=int, default=10,
                        help="Show N worst transcriptions")
    args = parser.parse_args()

    print("Loading model...", flush=True)
    model, config, tokenizer = load_model(args.model, args.config)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params:,} parameters", flush=True)

    print(f"Loading LibriSpeech {args.split}...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("openslr/librispeech_asr", split=args.split, streaming=True)

    references = []
    hypotheses = []
    errors = []  # (wer, ref, hyp) for worst-case analysis
    start = time.time()

    for i, sample in enumerate(ds):
        if args.max_samples and i >= args.max_samples:
            break

        # Get audio
        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio["sampling_rate"]

        # Get reference text (lowercase, strip)
        ref = sample["text"].lower().strip()

        # Make mel and decode
        mel = make_mel(waveform, sr, config)
        hyp = greedy_decode(model, mel, tokenizer).strip()

        references.append(ref)
        hypotheses.append(hyp)

        # Track per-sample WER
        sample_wer = wer(ref, hyp) if ref else 0.0
        errors.append((sample_wer, ref, hyp))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            current_wer = wer(references, hypotheses)
            print(f"  {i+1} samples | WER: {current_wer:.2%} | {elapsed:.0f}s", flush=True)

    total_time = time.time() - start
    total_wer = wer(references, hypotheses)
    total_cer = cer(references, hypotheses)

    print(f"\n{'='*60}")
    print(f"  LibriSpeech {args.split} — Greedy CTC Decode")
    print(f"{'='*60}")
    print(f"  Samples:    {len(references)}")
    print(f"  WER:        {total_wer:.2%}")
    print(f"  CER:        {total_cer:.2%}")
    print(f"  Time:       {total_time:.1f}s ({total_time/len(references):.2f}s/sample)")
    print(f"{'='*60}")

    # Show worst transcriptions
    if args.show_errors > 0:
        errors.sort(key=lambda x: -x[0])
        print(f"\n  Top {args.show_errors} worst transcriptions:")
        for j, (w, ref, hyp) in enumerate(errors[:args.show_errors]):
            print(f"\n  [{j+1}] WER: {w:.0%}")
            print(f"    GT:   '{ref}'")
            print(f"    PRED: '{hyp}'")


if __name__ == "__main__":
    main()
