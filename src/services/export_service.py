"""
ONNX Export Service.

Handles exporting Grassmann models to ONNX format for cross-platform inference.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.config import GrassmannConfig


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
    model = model.cpu()
    model.eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  Exporting model.onnx (CTC)...")
    wrapper = CTCModelWrapper(model)
    wrapper.eval()
    dummy_mel = torch.randn(1, config.n_mels, 200)
    torch.onnx.export(
        wrapper, (dummy_mel,), str(out / "model.onnx"),
        input_names=["mel"], output_names=["log_probs"],
        dynamic_axes={"mel": {0: "batch", 2: "time"}, "log_probs": {0: "batch", 1: "time"}},
        opset_version=opset,
    )


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
