#!/usr/bin/env python3
"""
Grassmann Transcription — Inference Script

Standalone inference for a trained Grassmann model.
No knowledge of the training pipeline required.

Usage:
    # From a saved model directory
    python inference.py --model checkpoints/my_model --audio speech.wav

    # From a HuggingFace Hub model
    python inference.py --model nfinch/grassmann-transcription --audio speech.wav

    # Quick test with built-in example
    python inference.py --model checkpoints/my_model --demo
"""

import argparse
import sys

import torch
import torchaudio
import torchaudio.transforms as T


def load_model(model_path: str, device: str = "cpu"):
    """Load model from a local directory or HuggingFace Hub."""
    from src.services.model_hub_service import load_model as _load
    return _load(model_path, device=device)


def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and preprocess an audio file into a mel spectrogram."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        waveform = T.Resample(sr, target_sr)(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr, n_fft=400, hop_length=160, n_mels=80,
    )
    mel = mel_transform(waveform)
    mel = torch.log(mel.clamp(min=1e-9))
    mel = mel.squeeze(0)  # (n_mels, time)

    # Normalize
    mel = (mel - mel.mean()) / mel.std().clamp(min=1e-9)

    return mel


def transcribe(model, config, tokenizer, mel: torch.Tensor, device: str = "cpu") -> str:
    """Run inference on a mel spectrogram and return text."""
    mel = mel.to(device)

    if config.use_seq2seq:
        # Autoregressive decoding
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        tokens = model.greedy_decode(mel, max_len=200)
    else:
        # CTC decoding
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        with torch.no_grad():
            log_probs = model(mel)  # (T, 1, vocab)
            tokens = log_probs.argmax(dim=-1).squeeze(1)

    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Grassmann Transcription — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --model checkpoints/best --audio recording.wav
  python inference.py --model checkpoints/best --audio recording.wav --device cuda
        """,
    )
    parser.add_argument("--model", required=True,
                        help="Path to saved model directory")
    parser.add_argument("--audio", type=str,
                        help="Path to audio file (.wav, .flac, .mp3)")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic audio to verify setup")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model, config, tokenizer = load_model(args.model, device=args.device)
    param_count = sum(p.numel() for p in model.parameters())
    mode = "seq2seq" if config.use_seq2seq else "CTC"
    print(f"  Loaded: {param_count:,} params, {mode} mode, {config.n_layers} encoder layers")

    if args.demo:
        # Synthetic audio test
        print("\nRunning demo with synthetic audio...")
        mel = torch.randn(config.n_mels, 200)
        text = transcribe(model, config, tokenizer, mel, args.device)
        print(f"  Output (untrained/random): '{text}'")
        print("  Setup verified!")
        return

    if not args.audio:
        parser.error("--audio is required (or use --demo)")

    # Transcribe
    print(f"Loading audio: {args.audio}")
    mel = load_audio(args.audio, target_sr=config.sample_rate)
    print(f"  Mel shape: {mel.shape} ({mel.shape[1] * config.hop_length / config.sample_rate:.1f}s)")

    text = transcribe(model, config, tokenizer, mel, args.device)
    print(f"\nTranscription:\n  {text}\n")


if __name__ == "__main__":
    main()
