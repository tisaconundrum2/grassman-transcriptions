"""
Evaluation Service.

WER/CER evaluation on LibriSpeech (or other HF datasets).
"""

from __future__ import annotations

import time

import torch
import torchaudio
import torchaudio.transforms

from Shared.models.config import GrassmannConfig
from Shared.services.tokenizer_service import CharTokenizer


_mel_transform = None


def make_mel(waveform: torch.Tensor, sample_rate: int, config: GrassmannConfig) -> torch.Tensor:
    """Convert waveform to normalized mel spectrogram (matches training pipeline)."""
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels,
        )

    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel = _mel_transform(waveform)
    mel = torch.log(mel.clamp(min=1e-9)).squeeze(0)

    mean = mel.mean()
    std = mel.std().clamp(min=1e-9)
    mel = (mel - mean) / std

    return mel.unsqueeze(0)


@torch.no_grad()
def greedy_decode(model, mel: torch.Tensor, tokenizer: CharTokenizer) -> str:
    log_probs = model(mel)
    preds = log_probs.argmax(dim=-1).squeeze(1)
    return tokenizer.decode(preds)


def evaluate_wer(
    model,
    config: GrassmannConfig,
    tokenizer: CharTokenizer,
    split: str = "test.clean",
    max_samples: int | None = None,
    show_errors: int = 10,
) -> dict:
    """Run WER/CER evaluation on a LibriSpeech split.

    Returns dict with wer, cer, num_samples, elapsed_seconds.
    """
    from jiwer import wer as compute_wer, cer as compute_cer
    from datasets import load_dataset

    model.eval()
    print(f"  Loading LibriSpeech {split}...")
    ds = load_dataset("openslr/librispeech_asr", split=split, streaming=True)

    references, hypotheses, errors = [], [], []
    start = time.time()

    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio["sampling_rate"]
        ref = sample["text"].lower().strip()

        mel = make_mel(waveform, sr, config)
        hyp = greedy_decode(model, mel, tokenizer).strip()

        references.append(ref)
        hypotheses.append(hyp)

        sample_wer = compute_wer(ref, hyp) if ref else 0.0
        errors.append((sample_wer, ref, hyp))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            current_wer = compute_wer(references, hypotheses)
            print(f"  {i+1} samples | WER: {current_wer:.2%} | {elapsed:.0f}s", flush=True)

    total_time = time.time() - start
    total_wer = compute_wer(references, hypotheses)
    total_cer = compute_cer(references, hypotheses)

    print(f"\n{'=' * 60}")
    print(f"  LibriSpeech {split} — Greedy CTC Decode")
    print(f"{'=' * 60}")
    print(f"  Samples:    {len(references)}")
    print(f"  WER:        {total_wer:.2%}")
    print(f"  CER:        {total_cer:.2%}")
    print(f"  Time:       {total_time:.1f}s ({total_time/len(references):.2f}s/sample)")
    print(f"{'=' * 60}")

    if show_errors > 0:
        errors.sort(key=lambda x: -x[0])
        print(f"\n  Top {show_errors} worst transcriptions:")
        for j, (w, ref, hyp) in enumerate(errors[:show_errors]):
            print(f"\n  [{j+1}] WER: {w:.0%}")
            print(f"    GT:   '{ref}'")
            print(f"    PRED: '{hyp}'")

    return {
        "wer": total_wer,
        "cer": total_cer,
        "num_samples": len(references),
        "elapsed_seconds": total_time,
    }
