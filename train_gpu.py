#!/usr/bin/env python3
"""
GPU Training Script for HuggingFace Spaces.

Runs a proper training run with:
  - Full LibriSpeech clean-100 (28K samples) or configurable subset
  - Cosine LR schedule with warmup
  - Periodic sample decoding to monitor transcription quality
  - Model checkpointing
  - CUDA/GPU auto-detection

Usage on HF Spaces (via Dockerfile CMD):
    python train_gpu.py

Local override:
    python train_gpu.py --max-samples 1000 --epochs 20 --batch-size 16
"""

import argparse
import os
import sys
import time

import torch

# Force unbuffered output so CI shows progress in real-time
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

from src.models.config import GrassmannConfig
from src.controllers.pipeline_controller import PipelineController
from src.controllers.model_controller import ModelController
from src.services.tokenizer_service import CharTokenizer
from src.services.hf_dataset_service import HFDatasetService


def _log_memory(label: str) -> None:
    """Log system and GPU memory usage."""
    try:
        import resource
        # maxrss is in bytes on macOS, KB on Linux
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux returns KB, macOS returns bytes
        if sys.platform == "darwin":
            rss_mb = rss_kb / 1024 / 1024
        else:
            rss_mb = rss_kb / 1024
        print(f"  [mem] {label}: RSS={rss_mb:.0f}MB", end="", flush=True)
    except Exception:
        print(f"  [mem] {label}: RSS=unknown", end="", flush=True)
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"  GPU_alloc={alloc:.0f}MB  GPU_reserved={reserved:.0f}MB", flush=True)
    else:
        print(flush=True)


def print_device_info() -> None:
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("  GPU: None (running on CPU)")


def decode_samples(model_ctrl: ModelController, dataset: HFDatasetService, n: int = 3) -> None:
    """Decode a few validation samples to show current transcription quality."""
    assert model_ctrl.trainer is not None
    assert dataset._val_ds is not None

    print("\n  Sample transcriptions:", flush=True)
    for i in range(min(n, len(dataset._val_ds))):
        mel, tokens = dataset._val_ds[i]
        # Decode prediction
        pred_text = model_ctrl.predict(mel)
        # Decode ground truth
        gt_text = model_ctrl.tokenizer.decode(tokens)
        print(f"    [{i}] GT:   '{gt_text}'", flush=True)
        print(f"    [{i}] PRED: '{pred_text}'")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grassmann GPU Training")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--split", default="train.clean.100")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples (None = full split, uses streaming if set)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--r-reduced", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--cross-window", type=int, default=8)
    parser.add_argument("--ctc", action="store_true",
                        help="Use CTC-only mode instead of seq2seq")
    parser.add_argument("--conformer", action="store_true",
                        help="Use Conformer blocks (ConvModule + Macaron FFN)")
    parser.add_argument("--two-sided", action="store_true",
                        help="Use bidirectional Grassmann mixing (for CTC/encoder)")
    parser.add_argument("--conv-kernel", type=int, default=31,
                        help="ConvModule kernel size")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--decode-every", type=int, default=5,
                        help="Decode sample transcriptions every N epochs")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Grassmann Transcription — GPU Training")
    print("=" * 60)
    print_device_info()
    print(f"  Dataset: {args.dataset} / {args.split}")
    print(f"  Max samples: {args.max_samples or 'full split'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    mode = "CTC (encoder-only)" if args.ctc else "Seq2Seq (encoder-decoder, fully attention-free)"
    block_type = "Conformer" if args.conformer else "Plain Grassmann"
    mix_dir = "two-sided" if args.two_sided else "causal"
    print(f"  Mode: {mode}")
    print(f"  Blocks: {block_type} ({mix_dir} mixing)")
    print(f"  Encoder: {args.layers}L / d={args.d_model} / r={args.r_reduced}")
    if args.conformer:
        print(f"  ConvModule kernel: {args.conv_kernel}")
    if not args.ctc:
        print(f"  Decoder: {args.decoder_layers}L / cross_window={args.cross_window}")
    print("=" * 60 + "\n")

    # Config
    config = GrassmannConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_layers=args.layers,
        n_decoder_layers=args.decoder_layers,
        cross_window=args.cross_window,
        d_model=args.d_model,
        r_reduced=args.r_reduced,
        learning_rate=args.lr,
        use_seq2seq=not args.ctc,
        use_conformer=args.conformer,
        two_sided=args.two_sided,
        conv_kernel_size=args.conv_kernel,
    )

    # Dataset — use disk cache for large datasets (>8K), in-memory for smaller
    cache_dir = os.path.join(args.save_dir, "mel_cache") if args.max_samples > 8000 else None
    tokenizer = CharTokenizer()
    dataset = HFDatasetService(config, tokenizer)
    dataset.load(
        args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        cache_dir=cache_dir,
    ).clean().preprocess().split()

    # Force cleanup after dataset caching (streaming buffers, librosa, etc.)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_memory("After dataset load")

    # Model
    model_ctrl = ModelController(config, tokenizer)
    model_ctrl.build()

    # LR scheduler
    assert model_ctrl.trainer is not None
    model_ctrl.trainer.set_cosine_schedule(args.epochs, warmup_epochs=min(3, args.epochs // 5))

    # DataLoaders
    train_loader = dataset.get_train_loader(config.batch_size)
    val_loader = dataset.get_val_loader(config.batch_size)

    # Training loop with periodic decoding
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    start_time = time.time()

    _log_memory("Before training")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = model_ctrl.trainer.train_epoch(train_loader)
        val_loss = model_ctrl.trainer.validate(val_loader)

        # Prevent CUDA memory fragmentation from killing us after many epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step LR scheduler
        scheduler = model_ctrl.trainer.scheduler
        if scheduler is not None:
            scheduler.step()

        lr = model_ctrl.trainer.optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        # Log GPU memory if available
        mem_str = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            mem_str = f"  GPU: {alloc:.1f}/{reserved:.1f}GB"

        print(f"  Epoch {epoch:3d}/{args.epochs}  |  "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}  "
              f"lr: {lr:.2e}  "
              f"({elapsed:.1f}s, total: {total_elapsed/60:.1f}min)"
              f"{mem_str}", flush=True)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(args.save_dir, "best_model.pt")
            assert model_ctrl.nn_model is not None
            torch.save(model_ctrl.nn_model.state_dict(), path)
            print(f"    → Saved best model (val_loss={val_loss:.4f})", flush=True)

        # Periodic checkpoint
        if epoch % 10 == 0:
            path = os.path.join(args.save_dir, f"epoch_{epoch}.pt")
            assert model_ctrl.nn_model is not None
            torch.save(model_ctrl.nn_model.state_dict(), path)

        # Decode samples to show progress
        if epoch % args.decode_every == 0 or epoch == 1:
            decode_samples(model_ctrl, dataset)

    # Final save
    final_path = os.path.join(args.save_dir, "final_model.pt")
    assert model_ctrl.nn_model is not None
    torch.save(model_ctrl.nn_model.state_dict(), final_path)

    # Test evaluation
    test_loader = dataset.get_test_loader(config.batch_size)
    test_loss = model_ctrl.trainer.validate(test_loader)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Test loss:     {test_loss:.4f}")
    print(f"  Total time:    {total_time/60:.1f} minutes")
    print(f"  Saved to:      {args.save_dir}/")
    print("=" * 60)

    # Final decode
    decode_samples(model_ctrl, dataset, n=5)

    # Save model bundle for export
    try:
        from src.services.model_hub_service import save_model
        assert model_ctrl.nn_model is not None
        save_model(model_ctrl.nn_model, config, os.path.join(args.save_dir, "bundle"))
        print("  Model bundle saved.", flush=True)
    except Exception as e:
        print(f"  Warning: model bundle save failed: {e}", flush=True)

    # Export ONNX if possible
    try:
        from export_onnx import export_seq2seq, export_ctc
        onnx_dir = os.path.join(args.save_dir, "onnx")
        if config.use_seq2seq:
            export_seq2seq(model_ctrl.nn_model, config, onnx_dir)
        else:
            export_ctc(model_ctrl.nn_model, config, onnx_dir)
        print(f"  ONNX exported to {onnx_dir}/", flush=True)
    except Exception as e:
        print(f"  Warning: ONNX export failed: {e}", flush=True)

    # Upload trained artifacts to HF repo (before pausing, or they're lost)
    try:
        from huggingface_hub import HfApi
        space_id = os.environ.get("SPACE_ID")
        if space_id:
            api = HfApi()
            bundle_dir = os.path.join(args.save_dir, "bundle")
            onnx_dir = os.path.join(args.save_dir, "onnx")
            if os.path.isdir(bundle_dir):
                api.upload_folder(
                    folder_path=bundle_dir,
                    path_in_repo="trained/bundle",
                    repo_id=space_id,
                    repo_type="space",
                )
                print(f"  Uploaded model bundle to {space_id}/trained/bundle/", flush=True)
            if os.path.isdir(onnx_dir):
                api.upload_folder(
                    folder_path=onnx_dir,
                    path_in_repo="trained/onnx",
                    repo_id=space_id,
                    repo_type="space",
                )
                print(f"  Uploaded ONNX to {space_id}/trained/onnx/", flush=True)
    except Exception as e:
        print(f"  Warning: artifact upload failed: {e}", flush=True)

    # Auto-pause HF Space to stop billing
    try:
        from huggingface_hub import HfApi
        space_id = os.environ.get("SPACE_ID")
        if space_id:
            api = HfApi()
            api.pause_space(space_id)
            print(f"  Space {space_id} paused to stop billing.", flush=True)
    except Exception as e:
        print(f"  Note: could not auto-pause Space: {e}", flush=True)


if __name__ == "__main__":
    main()
