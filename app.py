#!/usr/bin/env python3
"""
Grassmann Transcription — Unified CLI

One entry point for training, caching, evaluation, export, and inference.
Based on "Attention Is Not What You Need" (arXiv:2512.19428).

Usage:
    python app.py train --from-cache tisaconundrum/grassmann-mel-cache
    python app.py train --recipe 1750h
    python app.py cache --recipe 1750h --repo tisaconundrum/grassmann-mel-cache
    python app.py export --model checkpoints/bundle --output onnx_export/
    python app.py eval --model eval_model/best_model.pt --config eval_model/config.json
    python app.py transcribe --model checkpoints/bundle --audio speech.wav
    python app.py demo
"""

import argparse
import os
import sys
import time
import threading

import torch

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)


# ── Shared utilities ─────────────────────────────────────────────────────

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


def _log_memory(label: str) -> None:
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / (1024 * 1024 if sys.platform == "darwin" else 1024)
        print(f"  [mem] {label}: RSS={rss_mb:.0f}MB", end="", flush=True)
    except Exception:
        print(f"  [mem] {label}: RSS=unknown", end="", flush=True)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"  GPU_alloc={alloc:.0f}MB  GPU_reserved={reserved:.0f}MB", flush=True)
    else:
        print(flush=True)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--r-reduced", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--decoder-layers", type=int, default=4)
    p.add_argument("--cross-window", type=int, default=8)
    p.add_argument("--ctc", action="store_true", help="CTC-only mode")
    p.add_argument("--conformer", action="store_true", help="Conformer blocks")
    p.add_argument("--two-sided", action="store_true", help="Bidirectional mixing")
    p.add_argument("--conv-kernel", type=int, default=31)
    p.add_argument("--ablation", action="store_true", help="Symmetric-product ablation (no Grassmann geometry)")


def _auto_pause_space() -> None:
    try:
        from huggingface_hub import HfApi
        space_id = os.environ.get("SPACE_ID")
        if space_id:
            HfApi().pause_space(space_id)
            print(f"  Space {space_id} paused.", flush=True)
    except Exception as e:
        print(f"  Note: could not auto-pause: {e}", flush=True)


# ── train ────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> None:
    import gc
    from src.models.config import GrassmannConfig
    from src.controllers.model_controller import ModelController
    from src.services.tokenizer_service import CharTokenizer
    from src.repositories.dataset_repository import DatasetRepository
    from src.repositories.cache_repository import CacheRepository
    from src.services.dataset_service import DatasetService
    from src.services.checkpoint_service import CheckpointService

    _start_health_server()

    config = GrassmannConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        n_layers=args.layers, n_decoder_layers=args.decoder_layers,
        cross_window=args.cross_window, d_model=args.d_model,
        r_reduced=args.r_reduced, learning_rate=args.lr,
        use_seq2seq=not args.ctc, use_conformer=args.conformer,
        two_sided=args.two_sided, conv_kernel_size=args.conv_kernel,
        use_ablation=args.ablation,
    )

    print("\n" + "=" * 60)
    print("  Grassmann Transcription — Training")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("  GPU: None (CPU)")
    mode = "CTC" if args.ctc else "Seq2Seq"
    block = "Conformer" if args.conformer else "Plain"
    mix = "two-sided" if args.two_sided else "causal"
    mixer = "Ablation (symmetric)" if args.ablation else "Grassmann (Plücker)"
    print(f"  Mode: {mode} | Blocks: {block} ({mix})")
    print(f"  Mixer: {mixer}")
    print(f"  Encoder: {args.layers}L / d={args.d_model} / r={args.r_reduced}")
    print("=" * 60 + "\n")

    tokenizer = CharTokenizer()
    cache_dir = os.path.join(args.save_dir, "mel_cache")
    repo = DatasetRepository(config, tokenizer)
    cache = CacheRepository(cache_dir)
    dataset = DatasetService(repo, cache)

    if args.from_cache:
        dataset.load_from_hub(args.from_cache, max_samples=args.max_samples,
                              fetch_mode=args.fetch_mode).clean().preprocess().split()
    elif args.recipe:
        dataset.load_recipe(args.recipe,
                            max_samples_per_source=args.max_per_source).clean().preprocess().split()
    else:
        dataset.load_dataset(args.dataset, split=args.split,
                             max_samples=args.max_samples).clean().preprocess().split()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_memory("After dataset load")

    model_ctrl = ModelController(config, tokenizer)
    model_ctrl.build()
    assert model_ctrl.trainer is not None
    model_ctrl.trainer.set_cosine_schedule(args.epochs, warmup_epochs=min(3, args.epochs // 5))

    num_workers = getattr(args, 'workers', 0)
    train_loader = dataset.get_train_loader(config.batch_size, num_workers=num_workers)
    val_loader = dataset.get_val_loader(config.batch_size, num_workers=num_workers)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = 1
    start_time = time.time()

    # Resume from HF checkpoint if available
    ckpt_repo = args.checkpoint_repo
    if ckpt_repo is None and args.from_cache and not os.path.isdir(args.from_cache):
        ckpt_repo = args.from_cache
    ckpt_svc = CheckpointService(ckpt_repo, args.save_dir, args.run_name)
    assert model_ctrl.nn_model is not None
    start_epoch, best_val_loss = ckpt_svc.try_resume(model_ctrl.nn_model,
                                                     trainer=model_ctrl.trainer)

    _log_memory("Before training")

    for epoch in range(start_epoch, args.epochs + 1):
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

        print(f"  Epoch {epoch:3d}/{args.epochs}  |  "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
              f"lr: {lr:.2e}  ({elapsed:.1f}s, total: {total_elapsed/60:.1f}min)"
              f"{mem_str}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            assert model_ctrl.nn_model is not None
            torch.save(model_ctrl.nn_model.state_dict(),
                       os.path.join(args.save_dir, "best_model.pt"))
            print(f"    -> Saved best model (val_loss={val_loss:.4f})", flush=True)
            # Upload checkpoint to HF repo for crash resilience
            ckpt_svc.upload(epoch, val_loss, trainer=model_ctrl.trainer)

        if epoch % 10 == 0:
            assert model_ctrl.nn_model is not None
            torch.save(model_ctrl.nn_model.state_dict(),
                       os.path.join(args.save_dir, f"epoch_{epoch}.pt"))

        if epoch % args.decode_every == 0 or epoch == 1:
            model_ctrl.decode_samples(dataset._val_ds)

    assert model_ctrl.nn_model is not None
    torch.save(model_ctrl.nn_model.state_dict(), os.path.join(args.save_dir, "final_model.pt"))

    test_loader = dataset.get_test_loader(config.batch_size, num_workers=num_workers)
    test_loss = model_ctrl.trainer.validate(test_loader)
    total_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Test loss:     {test_loss:.4f}")
    print(f"  Total time:    {total_time/60:.1f} minutes")
    print(f"{'=' * 60}")

    model_ctrl.decode_samples(dataset._val_ds, n=5)
    model_ctrl.post_training(args.save_dir, run_name=args.run_name)


    _auto_pause_space()


# ── cache ────────────────────────────────────────────────────────────────

def cmd_cache(args: argparse.Namespace) -> None:
    from src.models.config import GrassmannConfig
    from src.services.tokenizer_service import CharTokenizer
    from src.repositories.dataset_repository import DatasetRepository
    from src.repositories.cache_repository import CacheRepository
    from src.services.dataset_service import DatasetService

    _start_health_server()

    import tempfile

    input_dir = getattr(args, 'input', None)
    mode = "local" if input_dir else "streaming"
    print("\n" + "=" * 60)
    print(f"  Grassmann — Dataset Caching ({mode})")
    print("=" * 60)
    print(f"  Recipe: {args.recipe}")
    print(f"  Output:  {args.repo}")
    if input_dir:
        print(f"  Input:   {input_dir}")
    print("=" * 60 + "\n")

    config = GrassmannConfig()
    tokenizer = CharTokenizer()
    repo = DatasetRepository(config, tokenizer, input_dir=input_dir)
    staging_dir = tempfile.mkdtemp(prefix="mel_staging_")
    cache = CacheRepository(staging_dir)
    dataset = DatasetService(repo, cache)

    start = time.time()
    dataset.load_recipe(args.recipe,
                        max_samples_per_source=args.max_per_source,
                        upload_repo=args.repo,
                        samples_per_shard=args.samples_per_shard,
                        workers=args.workers)
    import shutil
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"\n  Caching complete in {(time.time() - start)/60:.1f} minutes")

    _auto_pause_space()
    print(f"\n  Done! Now train with: python app.py train --from-cache {args.repo}")


# ── export ───────────────────────────────────────────────────────────────

def cmd_export(args: argparse.Namespace) -> None:
    import json
    from dataclasses import asdict
    from pathlib import Path
    from src.services.model_hub_service import load_model
    from src.services.export_service import export_seq2seq, export_ctc, verify_onnx

    print(f"\n  Loading model from {args.model}...")
    model, config, tokenizer = load_model(args.model, device="cpu")
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Mode: {'seq2seq' if config.use_seq2seq else 'CTC'}")

    if config.use_seq2seq:
        export_seq2seq(model, config, args.output, args.opset)
    else:
        export_ctc(model, config, args.output, args.opset)

    out = Path(args.output)
    with open(out / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(out / "tokenizer_info.json", "w") as f:
        json.dump({"type": "char", "vocab_size": tokenizer.vocab_size}, f, indent=2)

    print(f"\n  Export complete!")
    if args.verify:
        verify_onnx(args.output, config, config.use_seq2seq)


# ── eval ─────────────────────────────────────────────────────────────────

def cmd_eval(args: argparse.Namespace) -> None:
    import json
    from src.models.config import GrassmannConfig
    from src.models.grassmann_model import GrassmannTranscriptionModel
    from src.services.tokenizer_service import CharTokenizer
    from src.services.eval_service import evaluate_wer

    print("  Loading model...", flush=True)
    with open(args.config) as f:
        config_dict = json.load(f)
    config = GrassmannConfig(**{
        k: v for k, v in config_dict.items()
        if k in GrassmannConfig.__dataclass_fields__
    })
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    model = GrassmannTranscriptionModel(config)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    evaluate_wer(model, config, tokenizer, split=args.split,
                 max_samples=args.max_samples, show_errors=args.show_errors)


# ── transcribe ───────────────────────────────────────────────────────────

def cmd_transcribe(args: argparse.Namespace) -> None:
    from src.models.config import GrassmannConfig
    from src.controllers.pipeline_controller import PipelineController

    config = GrassmannConfig()
    pipeline = PipelineController(config)
    pipeline.load_model(args.model)
    text = pipeline.transcribe(args.audio)
    print(f"\n  {text}\n")


# ── demo ─────────────────────────────────────────────────────────────────

def cmd_demo() -> None:
    from src.models.config import GrassmannConfig
    from src.controllers.model_controller import ModelController
    from src.services.tokenizer_service import CharTokenizer

    print("\n  Running demo with synthetic data...\n")
    config = GrassmannConfig(n_layers=2, d_model=128, r_reduced=16, d_ff=512,
                              n_decoder_layers=2, epochs=3, batch_size=4)
    tokenizer = CharTokenizer()
    config.vocab_size = tokenizer.vocab_size

    model = ModelController(config, tokenizer)
    model.build()
    assert model.trainer is not None
    device = model.trainer.device
    mels = torch.randn(4, config.n_mels, 200, device=device)

    assert model.nn_model is not None
    if config.use_seq2seq:
        tokens = tokenizer.encode("hello world").unsqueeze(0).expand(4, -1)
        logits = model.nn_model(mels, tokens)
        print(f"  Forward pass OK — output: {logits.shape}")
    else:
        log_probs = model.nn_model(mels)
        print(f"  Forward pass OK — output: {log_probs.shape}")

    pred_tokens = model.trainer.greedy_decode(mels[0:1])
    decoded = tokenizer.decode(pred_tokens)
    print(f"  Greedy decode (untrained): '{decoded}'")
    print("\n  Demo complete!\n")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Grassmann Manifold Audio Transcription")
    sub = parser.add_subparsers(dest="command")

    # train
    p = sub.add_parser("train", help="Train model (GPU)")
    p.add_argument("--from-cache", default=None,
                   help="Pre-cached mel shards: HF Dataset repo id or local/mounted path")
    p.add_argument("--fetch-mode", default="auto",
                   choices=["auto", "local", "stream", "prefetch"],
                   help="Shard fetcher (auto = local if path exists, else prefetch)")
    p.add_argument("--checkpoint-repo", default=None,
                   help="HF Dataset repo for checkpoint upload/resume "
                        "(defaults to --from-cache when it's a repo id)")
    p.add_argument("--recipe", default=None, help="Multi-dataset recipe (e.g. '1750h')")
    p.add_argument("--dataset", default="openslr/librispeech_asr")
    p.add_argument("--split", default="train.clean.100")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-per-source", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--run-name", default=None, help="Prefix for uploaded artifacts (e.g. 'ablation-smol')")
    p.add_argument("--decode-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=0, help="DataLoader num_workers (0=main process)")
    _add_model_args(p)

    # cache
    p = sub.add_parser("cache", help="Cache dataset mels to HF Hub")
    p.add_argument("--recipe", required=True)
    p.add_argument("--repo", required=True, help="HF Dataset repo for mel shard upload")
    p.add_argument("--input", default=None, help="Local/mount dir with raw datasets (omit to stream from Hub)")
    p.add_argument("--max-per-source", type=int, default=None)
    p.add_argument("--samples-per-shard", type=int, default=5000, help="Samples per shard file (default 5000)")
    p.add_argument("--workers", type=int, default=0, help="Parallel mel workers (0=sequential, default)")

    # export
    p = sub.add_parser("export", help="Export model to ONNX")
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="onnx_export")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--verify", action="store_true")

    # eval
    p = sub.add_parser("eval", help="Evaluate WER/CER on LibriSpeech")
    p.add_argument("--model", default="eval_model/best_model.pt")
    p.add_argument("--config", default="eval_model/config.json")
    p.add_argument("--split", default="test.clean")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--show-errors", type=int, default=10)

    # transcribe
    p = sub.add_parser("transcribe", help="Transcribe audio file")
    p.add_argument("--model", required=True)
    p.add_argument("--audio", required=True)

    # demo
    sub.add_parser("demo", help="Sanity check with synthetic data")

    args = parser.parse_args()
    commands = {
        "train": cmd_train, "cache": cmd_cache, "export": cmd_export,
        "eval": cmd_eval, "transcribe": cmd_transcribe, "demo": lambda a: cmd_demo(),
    }
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
