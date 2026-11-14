#!/usr/bin/env python3
"""
ONNX Export for Grassmann Transcription Models.

Exports the model as ONNX for cross-platform inference without PyTorch.

Seq2Seq mode exports two models:
  - encoder.onnx:      mel → encoder_states  (one-shot)
  - decoder_step.onnx: (tokens, encoder_states) → logits  (called per step)

CTC mode exports one model:
  - model.onnx:        mel → log_probs

Usage:
    # Export from a saved model directory
    python export_onnx.py --model checkpoints/best --output onnx_model/

    # Export with custom opset
    python export_onnx.py --model checkpoints/best --output onnx_model/ --opset 17

    # Verify exported model
    python export_onnx.py --model checkpoints/best --output onnx_model/ --verify
"""

import argparse
import os
from pathlib import Path

import torch

from src.services.model_hub_service import load_model


class EncoderWrapper(torch.nn.Module):
    """Wraps the encoder for clean ONNX export."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)


class DecoderStepWrapper(torch.nn.Module):
    """Wraps the decoder for single-step ONNX export.

    Takes all generated tokens so far + encoder output,
    returns logits for the next token position only.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self, tokens: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        # Full forward through decoder (tokens = all generated so far)
        logits = self.decoder(tokens, encoder_out)  # (B, T, vocab)
        # Return only the last position's logits
        return logits[:, -1:, :]  # (B, 1, vocab)


class CTCModelWrapper(torch.nn.Module):
    """Wraps the CTC model for clean ONNX export (removes transpose)."""

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
        # Return (B, T, vocab) with log_softmax — no transpose
        return logits.log_softmax(dim=-1)


def export_seq2seq(model, config, output_dir: str, opset: int = 17) -> None:
    """Export encoder and decoder as separate ONNX models."""
    model.eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_mels = config.n_mels
    d_model = config.d_model

    # --- Encoder ---
    print("  Exporting encoder.onnx...")
    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()

    dummy_mel = torch.randn(1, n_mels, 200)

    torch.onnx.export(
        encoder_wrapper,
        (dummy_mel,),
        str(out / "encoder.onnx"),
        input_names=["mel"],
        output_names=["encoder_out"],
        dynamic_axes={
            "mel": {0: "batch", 2: "time"},
            "encoder_out": {0: "batch", 1: "enc_time"},
        },
        opset_version=opset,
    )
    print(f"    Saved: {out / 'encoder.onnx'}")

    # --- Decoder step ---
    print("  Exporting decoder_step.onnx...")
    decoder_wrapper = DecoderStepWrapper(model.decoder)
    decoder_wrapper.eval()

    # Dummy inputs: 3 tokens generated so far, encoder output of length 50
    dummy_tokens = torch.tensor([[0, 5, 12]], dtype=torch.long)  # (1, 3)
    dummy_enc_out = torch.randn(1, 50, d_model)

    torch.onnx.export(
        decoder_wrapper,
        (dummy_tokens, dummy_enc_out),
        str(out / "decoder_step.onnx"),
        input_names=["tokens", "encoder_out"],
        output_names=["next_logits"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq_len"},
            "encoder_out": {0: "batch", 1: "enc_time"},
            "next_logits": {0: "batch"},
        },
        opset_version=opset,
    )
    print(f"    Saved: {out / 'decoder_step.onnx'}")


def export_ctc(model, config, output_dir: str, opset: int = 17) -> None:
    """Export CTC model as a single ONNX model."""
    model.eval()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("  Exporting model.onnx (CTC)...")
    wrapper = CTCModelWrapper(model)
    wrapper.eval()

    dummy_mel = torch.randn(1, config.n_mels, 200)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(out / "model.onnx"),
        input_names=["mel"],
        output_names=["log_probs"],
        dynamic_axes={
            "mel": {0: "batch", 2: "time"},
            "log_probs": {0: "batch", 1: "time"},
        },
        opset_version=opset,
    )
    print(f"    Saved: {out / 'model.onnx'}")


def verify_onnx(output_dir: str, config, is_seq2seq: bool) -> None:
    """Verify exported ONNX models with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Install onnxruntime to verify: pip install onnxruntime")
        return

    import numpy as np

    out = Path(output_dir)

    if is_seq2seq:
        # Test encoder
        print("  Verifying encoder.onnx...")
        enc_session = ort.InferenceSession(str(out / "encoder.onnx"))
        mel = np.random.randn(1, config.n_mels, 200).astype(np.float32)
        enc_out = enc_session.run(None, {"mel": mel})[0]
        print(f"    Encoder output shape: {enc_out.shape}")

        # Test decoder step
        print("  Verifying decoder_step.onnx...")
        dec_session = ort.InferenceSession(str(out / "decoder_step.onnx"))
        tokens = np.array([[0, 5, 12]], dtype=np.int64)
        logits = dec_session.run(None, {
            "tokens": tokens,
            "encoder_out": enc_out,
        })[0]
        print(f"    Decoder logits shape: {logits.shape}")
        print(f"    Predicted token: {logits.argmax()}")

        # Full autoregressive loop
        print("  Running full autoregressive decode...")
        generated = [0]  # SOS
        for step in range(20):
            tok_input = np.array([generated], dtype=np.int64)
            step_logits = dec_session.run(None, {
                "tokens": tok_input,
                "encoder_out": enc_out,
            })[0]
            next_tok = int(step_logits[0, 0].argmax())
            generated.append(next_tok)
            if next_tok == 0:  # EOS
                break
        print(f"    Generated {len(generated)} tokens: {generated[:10]}...")
    else:
        print("  Verifying model.onnx (CTC)...")
        session = ort.InferenceSession(str(out / "model.onnx"))
        mel = np.random.randn(1, config.n_mels, 200).astype(np.float32)
        log_probs = session.run(None, {"mel": mel})[0]
        print(f"    Output shape: {log_probs.shape}")

    print("  ONNX verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Export Grassmann model to ONNX")
    parser.add_argument("--model", required=True, help="Path to saved model directory")
    parser.add_argument("--output", default="onnx_export", help="Output directory")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Verify with ONNX Runtime")
    args = parser.parse_args()

    print(f"\nLoading model from {args.model}...")
    model, config, tokenizer = load_model(args.model, device="cpu")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count:,} parameters")

    is_seq2seq = config.use_seq2seq
    print(f"  Mode: {'seq2seq' if is_seq2seq else 'CTC'}")
    print(f"  Exporting to {args.output}/\n")

    if is_seq2seq:
        export_seq2seq(model, config, args.output, args.opset)
    else:
        export_ctc(model, config, args.output, args.opset)

    # Copy config + tokenizer info for standalone use
    import json
    from dataclasses import asdict
    out = Path(args.output)
    with open(out / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(out / "tokenizer_info.json", "w") as f:
        json.dump({"type": "char", "vocab_size": tokenizer.vocab_size}, f, indent=2)

    print(f"\n  Export complete!")

    if args.verify:
        print()
        verify_onnx(args.output, config, is_seq2seq)

    # Print usage example
    print(f"\n  Usage with ONNX Runtime:")
    if is_seq2seq:
        print(f"    import onnxruntime as ort")
        print(f"    encoder = ort.InferenceSession('{args.output}/encoder.onnx')")
        print(f"    decoder = ort.InferenceSession('{args.output}/decoder_step.onnx')")
        print(f"    enc_out = encoder.run(None, {{'mel': mel_array}})[0]")
        print(f"    logits = decoder.run(None, {{'tokens': tok_array, 'encoder_out': enc_out}})[0]")
    else:
        print(f"    import onnxruntime as ort")
        print(f"    session = ort.InferenceSession('{args.output}/model.onnx')")
        print(f"    log_probs = session.run(None, {{'mel': mel_array}})[0]")
    print()


if __name__ == "__main__":
    main()
