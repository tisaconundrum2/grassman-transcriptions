"""Smoke test: run 5 epochs on 100 samples, then verify each epoch saved unique weights.

Usage: python smoke_test_checkpoints.py
Env: CODE_ROOT (dir containing Shared/), HF_TOKEN
"""

import os
import sys
import time

_root = os.environ.get("CODE_ROOT", "/app")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "Train.Ablation"))

import torch

from models.ablation_layer import AblationLayer
from models.ablation_conformer import AblationConformerBlock
from Shared.models.config import GrassmannConfig
from Shared.controllers.model_controller import ModelController
from Shared.services.tokenizer_service import CharTokenizer
from Shared.repositories.shard_repository import make_shard_repository
from Shared.services.dataset_service import ShardStreamingDataset, collate_fn
from Shared.services.checkpoint_service import CheckpointService

REPO = "tisaconundrum/grassmann-mel-cache"
SAVE_DIR = "/tmp/smoke_ckpt"
RUN_NAME = "smoke-test"
EPOCHS = 5
MAX_SAMPLES = 100

config = GrassmannConfig(epochs=EPOCHS, batch_size=16, n_layers=2, d_model=128, r_reduced=16,
                         learning_rate=1e-3, use_seq2seq=False, use_conformer=True,
                         two_sided=True, use_ablation=True)
ctrl = ModelController(config, CharTokenizer(), layer_cls=AblationLayer, block_cls=AblationConformerBlock)
ctrl.build()
tr = ctrl.trainer
tr.set_cosine_schedule(EPOCHS, warmup_epochs=1)

shard_repo = make_shard_repository(REPO, "stream")
shards = shard_repo.list_shards()
n = len(shards)
n_train = int(n * 0.8)
n_val = max(1, int(n * 0.1))
train_shards = shards[:n_train]
val_shards = shards[n_train:n_train + n_val]

train_ds = ShardStreamingDataset(shard_repo, train_shards[:1], total_samples=MAX_SAMPLES, shuffle=False, max_samples=MAX_SAMPLES)
val_ds = ShardStreamingDataset(shard_repo, val_shards[:1], total_samples=50, shuffle=False, max_samples=50)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)

# No HF upload for smoke test — just local saves
ckpt_svc = CheckpointService(None, SAVE_DIR, RUN_NAME)

checksums = {}
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss = tr.train_epoch(train_loader)
    val_loss = tr.validate(val_loader)
    if tr.scheduler:
        tr.scheduler.step()
    elapsed = time.time() - t0
    print(f"  Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)", flush=True)

    ckpt_svc.upload(epoch, val_loss, trainer=tr, model=ctrl.nn_model, config=config)

    # Read the saved file and checksum it
    saved = os.path.join(SAVE_DIR, "last_model.pt")
    if os.path.isfile(saved):
        sd = torch.load(saved, weights_only=True, map_location="cpu")
        cs = sum(v.double().abs().sum().item() for v in sd.values())
        checksums[epoch] = (cs, os.path.getsize(saved))
        print(f"    last_model.pt checksum={cs:.4f} size={os.path.getsize(saved)}", flush=True)
    else:
        print(f"    ERROR: last_model.pt not found!", flush=True)
        checksums[epoch] = (None, 0)

    # Also check archive
    archive_pt = os.path.join(SAVE_DIR, "archive", f"{RUN_NAME}-epoch_{epoch:03d}", "model.pt")
    if os.path.isfile(archive_pt):
        asd = torch.load(archive_pt, weights_only=True, map_location="cpu")
        acs = sum(v.double().abs().sum().item() for v in asd.values())
        print(f"    archive checksum={acs:.4f} matches_last={abs(acs - checksums[epoch][0]) < 1e-6}", flush=True)
    else:
        print(f"    archive model.pt not found!", flush=True)

print("\nRESULTS", flush=True)
unique = set()
all_ok = True
for ep, (cs, sz) in sorted(checksums.items()):
    dup = cs in unique if cs is not None else True
    if dup or cs is None:
        all_ok = False
    unique.add(cs)
    print(f"  epoch {ep}: checksum={cs:.4f if cs else 'MISSING'} size={sz} {'DUPLICATE' if dup else 'unique'}", flush=True)

print(f"\nVERDICT: {'PASS — all epochs have unique weights' if all_ok else 'FAIL — duplicate or missing weights'}", flush=True)
