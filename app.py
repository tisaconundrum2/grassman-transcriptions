#!/usr/bin/env python3
"""
Grassmann Transcription — Main Entry Point

Train a Grassmann manifold-based model (no attention!) for audio transcription.
Based on "Attention Is Not What You Need" (arXiv:2512.19428).

Usage:
    # Train on a HuggingFace dataset (recommended)
    python app.py train-hf --dataset openslr/librispeech_asr --split train.clean.100 --epochs 10

    # Quick test with limited samples
    python app.py train-hf --dataset openslr/librispeech_asr --split train.clean.100 --max-samples 100 --epochs 3

    # Train on a local dataset directory (expects metadata.csv + wavs/)
    python app.py train --data ./data/librispeech --epochs 30

    # Transcribe with a saved model
    python app.py transcribe --model model.pt --audio test.wav

    # Demo: sanity-check with synthetic data
    python app.py demo
"""

import argparse
import sys

import torch

from src.models.config import GrassmannConfig
from src.controllers.pipeline_controller import PipelineController
from src.controllers.model_controller import ModelController
from src.services.tokenizer_service import CharTokenizer


def demo() -> None:
    """Run a quick sanity check with synthetic data to verify the pipeline works."""
    print("\n  Running demo with synthetic data...\n")

    config = GrassmannConfig(
        n_layers=2,
        d_model=128,
        r_reduced=16,
        d_ff=512,
        epochs=3,
        batch_size=4,
    )

    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    # Build model
    _model = ModelController(config, tokenizer)
    _model.build()

    # Create synthetic mel spectrograms
    assert _model.trainer is not None
    device = _model.trainer.device
    batch_size = 4
    n_mels = config.n_mels
    time_steps = 200

    mels = torch.randn(batch_size, n_mels, time_steps, device=device)
    tokens = tokenizer.encode("hello world").unsqueeze(0).expand(batch_size, -1)
    mel_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    token_lengths = torch.full((batch_size,), tokens.shape[1], dtype=torch.long)

    # Quick forward pass test
    assert _model.nn_model is not None
    log_probs = _model.nn_model(mels)
    print(f"  Forward pass OK — output shape: {log_probs.shape}")
    print(f"  Expected: (time/4, batch={batch_size}, vocab={config.vocab_size})")

    # Quick decode test
    assert _model.trainer is not None
    pred_tokens = _model.trainer.greedy_decode(mels[0:1])
    decoded = tokenizer.decode(pred_tokens)
    print(f"  Greedy decode (untrained): '{decoded}'")

    print("\n  Demo complete! The Grassmann pipeline is working.\n")


def train(args: argparse.Namespace) -> None:
    """Train the model on a dataset."""
    config = GrassmannConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_layers=args.layers,
        d_model=args.d_model,
        r_reduced=args.r_reduced,
    )

    pipeline = PipelineController(config)
    pipeline.prepare_dataset(args.data).build_model().train(args.epochs)

    if args.save:
        pipeline.save_model(args.save)

    if args.transcribe:
        text = pipeline.transcribe(args.transcribe)
        print(f"\n  Transcription: {text}\n")


def train_hf(args: argparse.Namespace) -> None:
    """Train the model on a HuggingFace dataset."""
    config = GrassmannConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_layers=args.layers,
        d_model=args.d_model,
        r_reduced=args.r_reduced,
    )

    pipeline = PipelineController(config)
    pipeline.prepare_hf_dataset(
        args.dataset,
        config_name=args.config,
        split=args.split,
        audio_col=args.audio_col,
        text_col=args.text_col,
        max_samples=args.max_samples,
    ).build_model().train(args.epochs)

    if args.save:
        pipeline.save_model(args.save)

    if args.test:
        pipeline.test()


def transcribe(args: argparse.Namespace) -> None:
    """Transcribe an audio file with a saved model."""
    config = GrassmannConfig()
    pipeline = PipelineController(config)
    pipeline.load_model(args.model)
    text = pipeline.transcribe(args.audio)
    print(f"\n  Transcription: {text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grassmann Manifold Audio Transcription"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Demo
    subparsers.add_parser("demo", help="Quick sanity check with synthetic data")

    # Shared training args
    def add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--layers", type=int, default=6)
        p.add_argument("--d-model", type=int, default=256)
        p.add_argument("--r-reduced", type=int, default=32)
        p.add_argument("--save", type=str, help="Path to save model checkpoint")

    # Train on local data
    train_parser = subparsers.add_parser("train", help="Train on a local dataset directory")
    train_parser.add_argument("--data", required=True, help="Path to dataset directory")
    train_parser.add_argument("--transcribe", type=str, help="Audio file to transcribe after training")
    add_model_args(train_parser)

    # Train on HuggingFace dataset
    hf_parser = subparsers.add_parser("train-hf", help="Train on a HuggingFace dataset")
    hf_parser.add_argument("--dataset", required=True,
                           help="HF dataset name (e.g. openslr/librispeech_asr)")
    hf_parser.add_argument("--config", type=str, default=None,
                           help="Dataset config (e.g. 'clean', 'en')")
    hf_parser.add_argument("--split", type=str, default=None,
                           help="Split to load (e.g. 'train.clean.100')")
    hf_parser.add_argument("--audio-col", type=str, default=None,
                           help="Audio column name (auto-detected)")
    hf_parser.add_argument("--text-col", type=str, default=None,
                           help="Text column name (auto-detected)")
    hf_parser.add_argument("--max-samples", type=int, default=None,
                           help="Limit samples for quick experiments")
    hf_parser.add_argument("--test", action="store_true",
                           help="Run test evaluation after training")
    add_model_args(hf_parser)

    # Transcribe
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe audio file")
    trans_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    trans_parser.add_argument("--audio", required=True, help="Path to audio file")

    args = parser.parse_args()

    if args.command == "demo":
        demo()
    elif args.command == "train":
        train(args)
    elif args.command == "train-hf":
        train_hf(args)
    elif args.command == "transcribe":
        transcribe(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
