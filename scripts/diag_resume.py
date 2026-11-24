"""Resume exactly like Train.Ablation, train a few batches, report whether weights actually change."""

import os
import sys

import torch

_root = os.environ.get("CODE_ROOT", "/app")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "Train.Ablation"))

from models.ablation_layer import AblationLayer
from models.ablation_conformer import AblationConformerBlock
from Shared.models.config import GrassmannConfig
from Shared.controllers.model_controller import ModelController
from Shared.services.tokenizer_service import CharTokenizer
from Shared.services.checkpoint_service import CheckpointService
from Shared.repositories.shard_repository import make_shard_repository
from Shared.services.dataset_service import ShardStreamingDataset, collate_fn

REPO = "tisaconundrum/grassmann-mel-cache"
STEPS = int(os.environ.get("STEPS", "40"))

config = GrassmannConfig(epochs=50, batch_size=64, n_layers=6, d_model=256, r_reduced=32,
                         learning_rate=3e-4, use_seq2seq=False, use_conformer=True,
                         two_sided=True, use_ablation=True)
ctrl = ModelController(config, CharTokenizer(), layer_cls=AblationLayer, block_cls=AblationConformerBlock)
ctrl.build()
tr = ctrl.trainer
tr.set_cosine_schedule(50, warmup_epochs=3)


def checksum() -> float:
    return sum(p.detach().double().abs().sum().item() for p in ctrl.nn_model.parameters())


ckpt = CheckpointService(REPO, "/tmp/ckpt", "ablation-1750h-b128")
start_epoch, _ = ckpt.try_resume(ctrl.nn_model, trainer=tr)

model_ids = {id(p) for p in ctrl.nn_model.parameters()}
opt_ids = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
print(f"start_epoch={start_epoch} lr={tr.optimizer.param_groups[0]['lr']:.3e} "
      f"scaler_scale={tr.scaler.get_scale()} amp={tr.use_amp}", flush=True)
print(f"optimizer tracks model params: {model_ids == opt_ids} "
      f"(model={len(model_ids)} opt={len(opt_ids)} overlap={len(model_ids & opt_ids)})", flush=True)
print(f"requires_grad params: {sum(p.requires_grad for p in ctrl.nn_model.parameters())}/{len(model_ids)}", flush=True)

repo = make_shard_repository(REPO, "stream")
shards = repo.list_shards()
ds = ShardStreamingDataset(repo, shards[:1], total_samples=5000, shuffle=False)
loader = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=collate_fn)

before = checksum()
ctrl.nn_model.train()
skipped = 0
for i, (mels, tokens, ml, tl) in enumerate(loader):
    if i >= STEPS:
        break
    mels, tokens, ml, tl = mels.to(tr.device), tokens.to(tr.device), ml.to(tr.device), tl.to(tr.device)
    tr.optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=tr.use_amp):
        lp = ctrl.nn_model(mels)
        loss = tr.ctc_loss(lp, tokens, ctrl.nn_model.get_output_lengths(ml), tl)
    s0 = tr.scaler.get_scale()
    tr.scaler.scale(loss).backward()
    tr.scaler.unscale_(tr.optimizer)
    gn = torch.nn.utils.clip_grad_norm_(ctrl.nn_model.parameters(), max_norm=5.0)
    tr.scaler.step(tr.optimizer)
    tr.scaler.update()
    if tr.scaler.get_scale() < s0:
        skipped += 1
    if i % 10 == 0:
        print(f"  step {i:3d} loss={loss.item():.4f} grad_norm={gn.item():.3e} scale={tr.scaler.get_scale()}", flush=True)
after = checksum()
print(f"RESULT param_checksum before={before:.6f} after={after:.6f} changed={before != after} "
      f"skipped_steps={skipped}/{STEPS}", flush=True)
