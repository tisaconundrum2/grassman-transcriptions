#!/usr/bin/env python3
"""
Evaluate Model — WER/CER on LibriSpeech.

Loads any model type (ablation or grassmann) from config.json.
No flags. Config via env vars. Just: python app.py
"""

import json
import os
import sys

import torch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, _here)

from Shared.models.config import GrassmannConfig
from Shared.models.grassmann_model import GrassmannTranscriptionModel
from Shared.services.model_hub_service import resolve_layer_classes
from Shared.services.tokenizer_service import CharTokenizer
from services.eval_service import evaluate_wer


# ── Config (env vars) ────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "eval_model/best_model.pt")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "eval_model/config.json")
SPLIT = os.environ.get("SPLIT", "test.clean")
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None
SHOW_ERRORS = int(os.environ.get("SHOW_ERRORS", "10"))


def main() -> None:
    print("  Loading model...", flush=True)
    with open(CONFIG_PATH) as f:
        config_dict = json.load(f)
    config = GrassmannConfig(**{
        k: v for k, v in config_dict.items()
        if k in GrassmannConfig.__dataclass_fields__
    })
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    # Resolve the correct layer/block class from config.use_ablation
    classes = resolve_layer_classes(config)
    model = GrassmannTranscriptionModel(config,
                                         layer_cls=classes["layer_cls"],
                                         block_cls=classes["block_cls"])

    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # Remap legacy checkpoint keys
    remapped = {}
    for k, v in sd.items():
        nk = k.replace(".plucker.idx_", ".encoder.idx_") if ".plucker.idx_" in k else k
        nk = nk.replace(".encoder.idx_", ".plucker.idx_") if nk == k and ".encoder.idx_" in k else nk
        remapped[nk] = v
    model.load_state_dict(remapped if remapped != sd else sd)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    mixer = "Ablation (symmetric)" if getattr(config, "use_ablation", False) else "Grassmann (Plücker)"
    print(f"  Mixer: {mixer}")

    evaluate_wer(model, config, tokenizer, split=SPLIT,
                 max_samples=MAX_SAMPLES, show_errors=SHOW_ERRORS)


if __name__ == "__main__":
    main()
