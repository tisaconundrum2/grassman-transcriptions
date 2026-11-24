#!/usr/bin/env python3
"""
Train Grassmann Model — Plücker Coordinate Encoder.

Hardcoded: use_ablation=False, use_conformer=True, ctc=True, two_sided=True.
No flags. Config via env vars. Just: python app.py
"""

import gc
import os
import sys
import time
import threading

import torch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, _here)

from models.grassmann_layer import CausalGrassmannLayer
from models.grassmann_conformer import GrassmannConformerBlock
from Shared.models.config import GrassmannConfig
from Shared.controllers.model_controller import ModelController
from Shared.services.tokenizer_service import CharTokenizer
from Shared.repositories.dataset_repository import DatasetRepository
from Shared.repositories.cache_repository import CacheRepository
from Shared.services.dataset_service import DatasetService
from Shared.services.checkpoint_service import CheckpointService
from Shared.services import metrics_service


# ── Config (env vars) ────────────────────────────────────────────────────

CACHE_REPO = os.environ.get("CACHE_REPO", "tisaconundrum/grassmann-mel-cache")
CHECKPOINT_REPO = os.environ.get("CHECKPOINT_REPO", None)
RUN_NAME = os.environ.get("RUN_NAME", "grassmann")
EPOCHS = int(os.environ.get("EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None
FETCH_MODE = os.environ.get("FETCH_MODE", "auto")
SAVE_DIR = os.environ.get("SAVE_DIR", "/tmp/checkpoints")
DECODE_EVERY = int(os.environ.get("DECODE_EVERY", "5"))
WORKERS = int(os.environ.get("WORKERS", "0"))
LAYERS = int(os.environ.get("LAYERS", "6"))
D_MODEL = int(os.environ.get("D_MODEL", "256"))
R_REDUCED = int(os.environ.get("R_REDUCED", "32"))
LR = float(os.environ.get("LR", "3e-4"))


def _start_health_server(port: int = 7860) -> None:
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Running...")

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"  Health server on port {port}", flush=True)


def _auto_pause_space() -> None:
    try:
        from huggingface_hub import HfApi
        space_id = os.environ.get("SPACE_ID")
        if space_id:
            HfApi().pause_space(space_id)
            print(f"  Space {space_id} paused.", flush=True)
    except Exception as e:
        print(f"  Note: could not auto-pause: {e}", flush=True)


def preflight() -> None:
    """Synthetic data sanity check — verify model can forward-pass."""
    print("\n  Pre-flight check...", flush=True)
    config = GrassmannConfig(n_layers=2, d_model=128, r_reduced=16, d_ff=512,
                              use_seq2seq=False, use_conformer=True,
                              two_sided=True, use_ablation=False)
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size
    model = ModelController(config, tokenizer,
                            layer_cls=CausalGrassmannLayer, block_cls=GrassmannConformerBlock)
    model.build()
    assert model.nn_model is not None and model.trainer is not None
    mels = torch.randn(2, config.n_mels, 100, device=model.trainer.device)
    log_probs = model.nn_model(mels)
    print(f"  Pre-flight OK — output: {log_probs.shape}", flush=True)
    del model, mels, log_probs
    gc.collect()


def main() -> None:
    _start_health_server()
    preflight()

    config = GrassmannConfig(
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        n_layers=LAYERS, d_model=D_MODEL, r_reduced=R_REDUCED,
        learning_rate=LR,
        use_seq2seq=False, use_conformer=True,
        two_sided=True, use_ablation=False,
    )

    print("\n" + "=" * 60)
    print("  Grassmann Training — Plücker Coordinate Encoder")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("  GPU: None (CPU)")
    print(f"  Mode: CTC | Blocks: Conformer (two-sided)")
    print(f"  Mixer: Grassmann (Plücker)")
    print(f"  Encoder: {LAYERS}L / d={D_MODEL} / r={R_REDUCED}")
    print("=" * 60 + "\n")

    tokenizer = CharTokenizer()
    cache_dir = os.path.join(SAVE_DIR, "mel_cache")
    repo = DatasetRepository(config, tokenizer)
    cache = CacheRepository(cache_dir)
    dataset = DatasetService(repo, cache)

    dataset.load_from_hub(CACHE_REPO, max_samples=MAX_SAMPLES,
                          fetch_mode=FETCH_MODE).clean().preprocess().split()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_ctrl = ModelController(config, tokenizer,
                                 layer_cls=CausalGrassmannLayer, block_cls=GrassmannConformerBlock)
    model_ctrl.build()
    assert model_ctrl.trainer is not None
    model_ctrl.trainer.set_cosine_schedule(EPOCHS, warmup_epochs=min(3, EPOCHS // 5))

    train_loader = dataset.get_train_loader(config.batch_size, num_workers=WORKERS)
    val_loader = dataset.get_val_loader(config.batch_size, num_workers=WORKERS)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = 1
    start_time = time.time()

    ckpt_repo = CHECKPOINT_REPO or CACHE_REPO
    ckpt_svc = CheckpointService(ckpt_repo, SAVE_DIR, RUN_NAME)
    assert model_ctrl.nn_model is not None
    start_epoch, _ = ckpt_svc.try_resume(model_ctrl.nn_model, trainer=model_ctrl.trainer)
    best_val_loss = float("inf")

    model_ctrl.trainer.global_step = (start_epoch - 1) * len(train_loader)
    metrics_service.init(project="grassmann", name=RUN_NAME, config={
        "model": "grassmann", "layers": LAYERS, "d_model": D_MODEL, "r_reduced": R_REDUCED,
        "batch_size": BATCH_SIZE, "lr": LR, "epochs": EPOCHS, "start_epoch": start_epoch,
        "cache_repo": CACHE_REPO, "fetch_mode": FETCH_MODE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "image": os.environ.get("IMAGE_TAG", "unknown"),
    })

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        train_loss = model_ctrl.trainer.train_epoch(train_loader)
        val_loss = model_ctrl.trainer.validate(val_loader)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if model_ctrl.trainer.scheduler is not None:
            model_ctrl.trainer.scheduler.step()

        lr = model_ctrl.trainer.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        mem_str = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            mem_str = f"  GPU: {alloc:.1f}/{reserved:.1f}GB"

        print(f"  Epoch {epoch:3d}/{EPOCHS}  |  "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
              f"lr: {lr:.2e}  ({elapsed:.1f}s, total: {total_elapsed/60:.1f}min)"
              f"{mem_str}", flush=True)
        metrics_service.log({"epoch": epoch, "epoch/train_loss": train_loss,
                             "epoch/val_loss": val_loss, "epoch/lr": lr,
                             "epoch/minutes": elapsed / 60},
                            step=model_ctrl.trainer.global_step)

        ckpt_svc.upload(epoch, val_loss, trainer=model_ctrl.trainer,
                        model=model_ctrl.nn_model, config=config)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"    -> New best model (val_loss={val_loss:.4f})", flush=True)

        if epoch % DECODE_EVERY == 0 or epoch == 1:
            model_ctrl.decode_samples(dataset._val_ds)

    torch.save(model_ctrl.nn_model.state_dict(), os.path.join(SAVE_DIR, "final_model.pt"))

    test_loader = dataset.get_test_loader(config.batch_size, num_workers=WORKERS)
    test_loss = model_ctrl.trainer.validate(test_loader)
    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Test loss:     {test_loss:.4f}")
    print(f"  Total time:    {total_time/60:.1f} minutes")
    print(f"{'=' * 60}")
    metrics_service.log({"final/test_loss": test_loss, "final/best_val_loss": best_val_loss},
                        step=model_ctrl.trainer.global_step)
    metrics_service.finish()

    model_ctrl.decode_samples(dataset._val_ds, n=5)
    model_ctrl.post_training(SAVE_DIR, run_name=RUN_NAME)
    _auto_pause_space()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        import traceback
        metrics_service.alert(f"{RUN_NAME} crashed: {type(e).__name__}", traceback.format_exc()[-3000:])
        metrics_service.finish()
        raise
