"""Evaluate two checkpoints on the exact same validation shards.

Usage: compare_val.py <epoch_a> <epoch_b> [max_batches]
Separates "model got worse" from "validation set changed".
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Transcriptions.Resarch"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Transcriptions.Resarch", "Train.Ablation"))

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
EPOCH_A, EPOCH_B = int(sys.argv[1]), int(sys.argv[2])
MAX_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 40

shard_repo = make_shard_repository(REPO, "stream")
shards = shard_repo.list_shards()
n = len(shards)
n_train, n_val = int(n * 0.8), max(1, int(n * 0.1))
val_shards = shards[n_train:n_train + n_val]
print(f"{n} shards; validation shards = {val_shards[0]} .. {val_shards[-1]} ({len(val_shards)})")

config = GrassmannConfig(n_layers=6, d_model=256, r_reduced=32,
                         use_seq2seq=False, use_conformer=True,
                         two_sided=True, use_ablation=True)
tokenizer = CharTokenizer()
ctrl = ModelController(config, tokenizer, layer_cls=AblationLayer, block_cls=AblationConformerBlock)
ctrl.build()

ds = ShardStreamingDataset(shard_repo, val_shards[:1], total_samples=5000, shuffle=False)
batches = []
loader = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=collate_fn)
for i, b in enumerate(loader):
    if i >= MAX_BATCHES:
        break
    batches.append(b)
print(f"Cached {len(batches)} batches from {val_shards[0]}")


def evaluate(epoch: int) -> float:
    path = hf_hub_download(REPO, f"checkpoints/history/{RUN}-epoch_{epoch:03d}.pt", repo_type="dataset")
    sd = torch.load(path, weights_only=True, map_location="cpu")
    # Older checkpoints name the buffers `.plucker.`; current model uses `.encoder.`.
    sd = {k.replace(".plucker.idx_", ".encoder.idx_"): v for k, v in sd.items()}
    ctrl.nn_model.load_state_dict(sd)
    return ctrl.trainer.validate(batches)


for ep in (EPOCH_A, EPOCH_B):
    print(f"  epoch {ep:3d}: val_loss on identical batches = {evaluate(ep):.4f}")
