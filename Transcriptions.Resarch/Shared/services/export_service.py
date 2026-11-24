"""
ONNX Export Service.

Handles exporting Grassmann models to ONNX format for cross-platform inference.
Supports both mel-input models and end-to-end audio-input models with baked-in
mel spectrogram computation.
"""

from __future__ import annotations

import math
import copy
from pathlib import Path

import torch

from Shared.models.config import GrassmannConfig


class OnnxMelSpectrogram(torch.nn.Module):
    """ONNX-exportable mel spectrogram matching the .NET MelTransform.

    Computes: audio → STFT(Hann, zero-pad to pow2) → power[0:n_fft//2+1]
              → mel filterbank → log(clamp(1e-9)) → z-normalize.

    All ops are pure torch — no torchaudio, no numpy, no external libs.
    This is the canonical mel implementation; .NET, JS, and Python inference
    must all produce identical output for the same audio input.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 400,
                 hop_length: int = 160, n_mels: int = 80):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        fft_size = 1 << (n_fft - 1).bit_length()  # next power of 2
        self.fft_size = fft_size
        n_bins = n_fft // 2 + 1  # 201 — matches .NET (not fft_size//2+1)

        # Hann window (periodic) — registered as buffer so it's in the ONNX graph
        window = torch.hann_window(n_fft, periodic=True)
        self.register_buffer("window", window)

        # Mel filterbank — precomputed, frozen in the graph
        fb = self._build_filterbank(n_mels, n_fft, n_bins, sample_rate)
        self.register_buffer("filterbank", fb)  # (n_mels, n_bins)

    @staticmethod
    def _build_filterbank(n_mels: int, n_fft: int, n_bins: int,
                          sample_rate: int) -> torch.Tensor:
        """HTK mel filterbank with floor() bin mapping, (n_fft+1) scaling."""
        f_max = sample_rate / 2.0
        mel_min = 2595.0 * math.log10(1.0)
        mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)

        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

        fb = torch.zeros(n_mels, n_bins)
        for m in range(n_mels):
            left, center, right = bin_points[m], bin_points[m + 1], bin_points[m + 2]
            for k in range(left, min(center, n_bins)):
                if center > left:
                    fb[m, k] = float(k - left) / float(center - left)
            for k in range(center, min(right, n_bins)):
                if right > center:
                    fb[m, k] = float(right - k) / float(right - center)
        return fb

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (batch, samples) → mel: (batch, n_mels, frames), z-normalized."""
        # STFT: n_fft=fft_size (512) handles the zero-padding internally.
        # win_length=n_fft (400) with the Hann window matches .NET/JS framing.
        spec = torch.stft(
            audio, n_fft=self.fft_size, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window,
            center=False, return_complex=True,
        )  # (batch, fft_size//2+1, frames)

        # Take only first n_fft//2+1 bins (201) — discard zero-pad frequencies
        n_bins = self.n_fft // 2 + 1
        spec = spec[:, :n_bins, :]  # (batch, 201, frames)

        # Power spectrum → mel → log
        power = spec.real ** 2 + spec.imag ** 2  # (batch, 201, frames)
        mel = torch.matmul(self.filterbank, power)  # (batch, n_mels, frames)
        mel = torch.log(mel.clamp(min=1e-9))

        # Per-utterance z-normalization
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True).clamp(min=1e-9)
        mel = (mel - mean) / std

        return mel


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)


class DecoderStepWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(tokens, encoder_out)
        return logits[:, -1:, :]


class CTCModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.frontend = model.frontend
        self.pos_enc = model.pos_enc
        self.layers = model.layers
        self.output_proj = model.output_proj

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.frontend(mel)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.output_proj(x)
        return logits.log_softmax(dim=-1)


class EndToEndCTCWrapper(torch.nn.Module):
    """Audio-in, log-probs-out. Mel spectrogram baked into the ONNX graph."""

    def __init__(self, model, config: GrassmannConfig):
        super().__init__()
        self.mel = OnnxMelSpectrogram(
            sample_rate=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels,
        )
        self.frontend = model.frontend
        self.pos_enc = model.pos_enc
        self.layers = model.layers
        self.output_proj = model.output_proj

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (batch, samples) → log_probs: (batch, time, vocab)"""
        mel = self.mel(audio)       # (batch, n_mels, frames)
        x = self.frontend(mel)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.output_proj(x)
        return logits.log_softmax(dim=-1)


def export_seq2seq(model, config: GrassmannConfig, output_dir: str, opset: int = 18) -> None:
    model = model.cpu()
    model.eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  Exporting encoder.onnx...")
    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()
    dummy_mel = torch.randn(1, config.n_mels, 200)
    torch.onnx.export(
        encoder_wrapper, (dummy_mel,), str(out / "encoder.onnx"),
        input_names=["mel"], output_names=["encoder_out"],
        dynamic_axes={"mel": {0: "batch", 2: "time"}, "encoder_out": {0: "batch", 1: "enc_time"}},
        opset_version=opset,
    )

    print("  Exporting decoder_step.onnx...")
    decoder_wrapper = DecoderStepWrapper(model.decoder)
    decoder_wrapper.eval()
    dummy_tokens = torch.tensor([[0, 5, 12]], dtype=torch.long)
    dummy_enc_out = torch.randn(1, 50, config.d_model)
    torch.onnx.export(
        decoder_wrapper, (dummy_tokens, dummy_enc_out), str(out / "decoder_step.onnx"),
        input_names=["tokens", "encoder_out"], output_names=["next_logits"],
        dynamic_axes={"tokens": {0: "batch", 1: "seq_len"}, "encoder_out": {0: "batch", 1: "enc_time"}, "next_logits": {0: "batch"}},
        opset_version=opset,
    )


def export_ctc(model, config: GrassmannConfig, output_dir: str, opset: int = 18) -> None:
    model = copy.deepcopy(model).cpu()
    model.eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  Exporting model.onnx (CTC, mel input)...")
    wrapper = CTCModelWrapper(model)
    wrapper.eval()
    dummy_mel = torch.randn(1, config.n_mels, 200)
    torch.onnx.export(
        wrapper, (dummy_mel,), str(out / "model.onnx"),
        input_names=["mel"], output_names=["log_probs"],
        dynamic_axes={"mel": {0: "batch", 2: "time"}, "log_probs": {0: "batch", 1: "time"}},
        opset_version=opset,
        external_data=False,  # single self-contained .onnx (needed for browser upload)
    )


def export_ctc_e2e(model, config: GrassmannConfig, output_dir: str, opset: int = 18) -> None:
    """Export end-to-end CTC model: raw audio in → log_probs out.

    The mel spectrogram is baked into the ONNX graph using OnnxMelSpectrogram,
    which matches the .NET MelTransform exactly. Consumers pass raw 16kHz
    mono float32 audio and get CTC log-probs back — no mel code needed.
    """
    model = copy.deepcopy(model).cpu()
    model.eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  Exporting model_e2e.onnx (CTC, audio input, mel baked in)...")
    wrapper = EndToEndCTCWrapper(model, config)
    wrapper.eval()
    dummy_audio = torch.randn(1, config.sample_rate * 3)  # 3 seconds
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy_audio,), str(out / "model_e2e.onnx"),
            input_names=["audio"], output_names=["log_probs"],
            dynamic_axes={
                "audio": {0: "batch", 1: "samples"},
                "log_probs": {0: "batch", 1: "time"},
            },
            opset_version=opset,
            external_data=False,
        )
    print(f"    Input:  audio (batch, samples) — 16kHz mono float32")
    print(f"    Output: log_probs (batch, time, {config.vocab_size})")


def verify_onnx(output_dir: str, config: GrassmannConfig, is_seq2seq: bool) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Install onnxruntime to verify: pip install onnxruntime")
        return

    import numpy as np
    out = Path(output_dir)

    if is_seq2seq:
        print("  Verifying encoder.onnx...")
        enc_session = ort.InferenceSession(str(out / "encoder.onnx"))
        mel = np.random.randn(1, config.n_mels, 200).astype(np.float32)
        enc_out = enc_session.run(None, {"mel": mel})[0]
        print(f"    Encoder output shape: {enc_out.shape}")

        print("  Verifying decoder_step.onnx...")
        dec_session = ort.InferenceSession(str(out / "decoder_step.onnx"))
        tokens = np.array([[0, 5, 12]], dtype=np.int64)
        logits = dec_session.run(None, {"tokens": tokens, "encoder_out": enc_out})[0]
        print(f"    Decoder logits shape: {logits.shape}")
    else:
        print("  Verifying model.onnx (CTC)...")
        session = ort.InferenceSession(str(out / "model.onnx"))
        mel = np.random.randn(1, config.n_mels, 200).astype(np.float32)
        log_probs = session.run(None, {"mel": mel})[0]
        print(f"    Output shape: {log_probs.shape}")

    print("  ONNX verification passed!")
