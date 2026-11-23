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
import os
from dataclasses import asdict
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn

from Shared.models.config import GrassmannConfig
from Shared.models.grassmann_model import GrassmannTranscriptionModel
from Shared.services.tokenizer_service import CharTokenizer


def resolve_layer_classes(config: GrassmannConfig) -> dict:
    """Resolve the correct layer/block class based on config.use_ablation.

    Returns dict with 'layer_cls' and 'block_cls' keys.
    Uses importlib to handle dotted directory names (Train.Ablation, Train.Grassman).
    """
    import importlib
    import importlib.util

    ablation = getattr(config, "use_ablation", False)

    def _import_from_file(module_name: str, search_paths: list[str]):
        """Try to import a module from multiple possible file paths."""
        for path in search_paths:
            if os.path.isfile(path):
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod
        raise ImportError(f"Cannot find {module_name} in {search_paths}")

    # Resolve paths relative to this file's grandparent (Transcriptions.Resarch/)
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if ablation:
        layer_mod = _import_from_file("ablation_layer", [
            os.path.join(base, "Train.Ablation", "models", "ablation_layer.py"),
        ])
        block_mod = _import_from_file("ablation_conformer", [
            os.path.join(base, "Train.Ablation", "models", "ablation_conformer.py"),
        ])
        return {"layer_cls": layer_mod.AblationLayer, "block_cls": block_mod.AblationConformerBlock}
    else:
        layer_mod = _import_from_file("grassmann_layer", [
            os.path.join(base, "Train.Grassman", "models", "grassmann_layer.py"),
        ])
        block_mod = _import_from_file("grassmann_conformer", [
            os.path.join(base, "Train.Grassman", "models", "grassmann_conformer.py"),
        ])
        return {"layer_cls": layer_mod.CausalGrassmannLayer, "block_cls": block_mod.GrassmannConformerBlock}


def save_model(
    model: nn.Module,
    config: GrassmannConfig,
    save_dir: str,
) -> None:
    """Save model + config in a self-contained directory."""
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    with open(path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    torch.save(model.state_dict(), path / "model.pt")

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
    layer_cls: Type[nn.Module] | None = None,
    block_cls: Type[nn.Module] | None = None,
) -> tuple[GrassmannTranscriptionModel, GrassmannConfig, CharTokenizer]:
    """Load a saved model from directory. Returns (model, config, tokenizer).

    If layer_cls/block_cls not provided, resolves from config.use_ablation.
    """
    path = Path(load_dir)

    with open(path / "config.json") as f:
        config_dict = json.load(f)
    config = GrassmannConfig(**{
        k: v for k, v in config_dict.items()
        if k in GrassmannConfig.__dataclass_fields__
    })

    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    if layer_cls is None or block_cls is None:
        resolved = resolve_layer_classes(config)
        layer_cls = layer_cls or resolved["layer_cls"]
        block_cls = block_cls or resolved["block_cls"]

    model = GrassmannTranscriptionModel(config, layer_cls=layer_cls, block_cls=block_cls)

    state = torch.load(path / "model.pt", map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        remapped = {
            k.replace(".plucker.idx_i", ".encoder.idx_i")
             .replace(".plucker.idx_j", ".encoder.idx_j"): v
            for k, v in state.items()
        }
        try:
            model.load_state_dict(remapped)
        except RuntimeError:
            remapped = {
                k.replace(".encoder.idx_i", ".plucker.idx_i")
                 .replace(".encoder.idx_j", ".plucker.idx_j"): v
                for k, v in state.items()
            }
            model.load_state_dict(remapped)
    model.to(device)
    model.eval()

    return model, config, tokenizer
