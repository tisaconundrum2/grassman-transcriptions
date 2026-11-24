"""Evaluate several checkpoints on the exact same validation batches.

Usage: compare_val.py <epoch> [<epoch> ...]
Env: MAX_BATCHES (0 = all), VAL_SHARDS (default 1), CODE_ROOT (dir containing Shared/).
Separates "model got worse" from "validation set changed".
"""

import os
import sys

import torch

_root = os.environ.get("CODE_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Transcriptions.Resarch")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "Train.Ablation"))

from huggingface_hub import hf_hub_download

from models.ablation_layer import AblationLayer
from models.ablation_conformer import AblationConformerBlock
from Shared.models.config import GrassmannConfig
from Shared.controllers.model_controller import ModelController
from Shared.services.tokenizer_service import CharTokenizer
from Shared.repositories.shard_repository import make_shard_repository
from Shared.services.dataset_service import ShardStreamingDataset, collate_fn

REPO = "tisaconundrum/grassmann-mel-cache"
RUN = "ablation-1750h-b128"
EPOCHS = [int(e) for e in sys.argv[1:]]
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
VAL_SHARDS = int(os.environ.get("VAL_SHARDS", "1"))

shard_repo = make_shard_repository(REPO, "stream")
shards = shard_repo.list_shards()
n = len(shards)
n_train, n_val = int(n * 0.8), max(1, int(n * 0.1))
val_shards = shards[n_train:n_train + n_val][:VAL_SHARDS]
print(f"{n} shards; scoring on {val_shards}", flush=True)

config = GrassmannConfig(n_layers=6, d_model=256, r_reduced=32,
                         use_seq2seq=False, use_conformer=True,
                         two_sided=True, use_ablation=True)
ctrl = ModelController(config, CharTokenizer(), layer_cls=AblationLayer, block_cls=AblationConformerBlock)
ctrl.build()

ds = ShardStreamingDataset(shard_repo, val_shards, total_samples=5000 * len(val_shards), shuffle=False)
batches = []
for i, b in enumerate(torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=collate_fn)):
    if MAX_BATCHES and i >= MAX_BATCHES:
        break
    batches.append(b)
print(f"Cached {len(batches)} batches on {ctrl.trainer.device}", flush=True)


def evaluate(epoch: int) -> float:
    path = hf_hub_download(REPO, f"checkpoints/history/{RUN}-epoch_{epoch:03d}.pt", repo_type="dataset")
    sd = torch.load(path, weights_only=True, map_location="cpu")
    # Older checkpoints name the buffers `.plucker.`; current model uses `.encoder.`.
    sd = {k.replace(".plucker.idx_", ".encoder.idx_"): v for k, v in sd.items()}
    ctrl.nn_model.load_state_dict(sd)
    return ctrl.trainer.validate(batches)


print("RESULTS", flush=True)
for ep in EPOCHS:
    print(f"  epoch {ep:3d}: val_loss on identical batches = {evaluate(ep):.4f}", flush=True)
