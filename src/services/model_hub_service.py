"""
Model Hub Service — Save/load models in a self-contained format.

Bundles config + weights + tokenizer info into a single directory
that anyone can download and run inference with.

Saved format:
    model_dir/
        config.json          # GrassmannConfig as JSON
        model.pt             # state_dict weights
        tokenizer_info.json  # vocab info for reconstruction
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from src.models.config import GrassmannConfig
from src.models.grassmann_seq2seq import GrassmannSeq2SeqModel
from src.models.grassmann_model import GrassmannTranscriptionModel
from src.services.tokenizer_service import CharTokenizer


def save_model(
    model: GrassmannSeq2SeqModel | GrassmannTranscriptionModel,
    config: GrassmannConfig,
    save_dir: str,
) -> None:
    """Save model + config in a self-contained directory."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = asdict(config)
    with open(path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save weights
    torch.save(model.state_dict(), path / "model.pt")

    # Save tokenizer info
    tokenizer = CharTokenizer()
    tok_info = {
        "type": "char",
        "vocab_size": tokenizer.vocab_size,
        "blank_id": tokenizer.blank_id,
    }
    with open(path / "tokenizer_info.json", "w") as f:
        json.dump(tok_info, f, indent=2)

    print(f"  Model saved to {save_dir}/")
    print(f"    config.json, model.pt, tokenizer_info.json")


def load_model(
    load_dir: str,
    device: str = "cpu",
) -> tuple[GrassmannSeq2SeqModel | GrassmannTranscriptionModel, GrassmannConfig, CharTokenizer]:
    """Load a saved model from directory. Returns (model, config, tokenizer)."""
    path = Path(load_dir)

    # Load config
    with open(path / "config.json") as f:
        config_dict = json.load(f)
    config = GrassmannConfig(**config_dict)

    # Build model
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    if config.use_seq2seq:
        model = GrassmannSeq2SeqModel(config)
    else:
        model = GrassmannTranscriptionModel(config)

    # Load weights
    state = torch.load(path / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, config, tokenizer
