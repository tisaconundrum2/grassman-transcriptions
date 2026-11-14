#!/usr/bin/env python3
"""
ONNX Inference — No PyTorch Required.

Standalone transcription using only ONNX Runtime + soundfile.
This is what end users run — no PyTorch, no training code needed.

Dependencies: onnxruntime, soundfile, numpy

Usage:
    pip install onnxruntime soundfile numpy
    python inference_onnx.py --model onnx_export/ --audio speech.wav
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio and compute mel spectrogram using only numpy + soundfile."""
    import soundfile as sf

    waveform, sr = sf.read(audio_path, dtype="float32")

    # Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Simple resampling via linear interpolation if needed
    if sr != target_sr:
        duration = len(waveform) / sr
        n_samples = int(duration * target_sr)
        indices = np.linspace(0, len(waveform) - 1, n_samples)
        waveform = np.interp(indices, np.arange(len(waveform)), waveform)

    return waveform


def mel_spectrogram_np(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute log-mel spectrogram using numpy (no torch/librosa)."""
    # STFT
    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    frames = np.stack([
        waveform[i * hop_length : i * hop_length + n_fft]
        for i in range(n_frames)
    ])
    window = np.hanning(n_fft)
    frames = frames * window
    stft = np.fft.rfft(frames, n=n_fft)
    power = np.abs(stft) ** 2

    # Mel filterbank
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    mel = filterbank @ power.T  # (n_mels, n_frames)
    mel = np.log(np.maximum(mel, 1e-9))

    # Normalize
    mean = mel.mean()
    std = max(mel.std(), 1e-9)
    mel = (mel - mean) / std

    return mel.astype(np.float32)


class CharTokenizerONNX:
    """Minimal char tokenizer for ONNX inference — no PyTorch needed."""

    def __init__(self):
        self._chars = " " + "abcdefghijklmnopqrstuvwxyz" + "'"
        self._idx_to_char = {i + 1: c for i, c in enumerate(self._chars)}
        self._idx_to_char[0] = ""

    def decode(self, token_ids: list[int]) -> str:
        chars = []
        prev = -1
        for idx in token_ids:
            if idx == 0:  # blank/eos
                prev = idx
                continue
            if idx != prev:
                chars.append(self._idx_to_char.get(idx, ""))
            prev = idx
        return "".join(chars)


def transcribe_seq2seq(
    encoder_path: str,
    decoder_path: str,
    mel: np.ndarray,
    max_len: int = 200,
) -> list[int]:
    """Autoregressive decoding with ONNX Runtime."""
    import onnxruntime as ort

    enc_session = ort.InferenceSession(encoder_path)
    dec_session = ort.InferenceSession(decoder_path)

    # Encode: (1, n_mels, time)
    mel_input = mel[np.newaxis, :, :]  # add batch dim
    enc_out = enc_session.run(None, {"mel": mel_input})[0]

    # Decode autoregressively
    generated = [0]  # SOS
    for _ in range(max_len):
        tokens = np.array([generated], dtype=np.int64)
        logits = dec_session.run(None, {
            "tokens": tokens,
            "encoder_out": enc_out,
        })[0]  # (1, 1, vocab)

        next_token = int(logits[0, 0].argmax())
        generated.append(next_token)

        if next_token == 0:  # EOS
            break

    return generated[1:]  # drop SOS


def transcribe_ctc(
    model_path: str,
    mel: np.ndarray,
) -> list[int]:
    """CTC greedy decoding with ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(model_path)
    mel_input = mel[np.newaxis, :, :]
    log_probs = session.run(None, {"mel": mel_input})[0]  # (1, T, vocab)

    # Greedy: argmax per frame
    return log_probs[0].argmax(axis=-1).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Grassmann Transcription — ONNX Inference (no PyTorch needed)",
    )
    parser.add_argument("--model", required=True, help="Path to ONNX export directory")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    args = parser.parse_args()

    model_dir = Path(args.model)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    is_seq2seq = config.get("use_seq2seq", True)
    tokenizer = CharTokenizerONNX()

    # Load and process audio
    print(f"Loading audio: {args.audio}")
    waveform = load_audio(args.audio, target_sr=config["sample_rate"])
    mel = mel_spectrogram_np(
        waveform,
        sr=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        n_mels=config["n_mels"],
    )
    print(f"  Mel shape: {mel.shape} ({len(waveform) / config['sample_rate']:.1f}s)")

    # Transcribe
    print("Transcribing...")
    if is_seq2seq:
        tokens = transcribe_seq2seq(
            str(model_dir / "encoder.onnx"),
            str(model_dir / "decoder_step.onnx"),
            mel,
        )
    else:
        tokens = transcribe_ctc(str(model_dir / "model.onnx"), mel)

    text = tokenizer.decode(tokens)
    print(f"\nTranscription:\n  {text}\n")


if __name__ == "__main__":
    main()
