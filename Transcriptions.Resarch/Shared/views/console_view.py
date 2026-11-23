"""Console view: prints training progress and results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Shared.models.config import GrassmannConfig
    from Shared.services.training_service import TrainMetrics


class ConsoleView:
    """Simple console-based display for training progress."""

    def show_model_info(self, config: GrassmannConfig, param_count: int) -> None:
        mode = "Seq2Seq (Encoder-Decoder)" if config.use_seq2seq else "CTC (Encoder-only)"
        print("\n" + "=" * 60)
        print(f"  Grassmann Transcription Model — {mode}")
        print("=" * 60)
        print(f"  Encoder layers: {config.n_layers}")
        if config.use_seq2seq:
            print(f"  Decoder layers: {config.n_decoder_layers}")
            print(f"  Cross window:   {config.cross_window}")
        print(f"  d_model:        {config.d_model}")
        print(f"  r_reduced:      {config.r_reduced}")
        print(f"  Plücker dim:    {config.plucker_dim}")
        print(f"  Window offsets: {config.window_offsets}")
        print(f"  Vocab size:     {config.vocab_size}")
        print(f"  Parameters:     {param_count:,}")
        print(f"  Attention:      NONE (fully Grassmann)")
        print("=" * 60 + "\n")

    def on_epoch_end(self, metrics: TrainMetrics) -> None:
        val_str = f"  val_loss: {metrics.val_loss:.4f}" if metrics.val_loss is not None else ""
        print(f"  Epoch {metrics.epoch:3d}  |  train_loss: {metrics.train_loss:.4f}{val_str}")

    def show_training_complete(self, history: list[TrainMetrics]) -> None:
        if history:
            best = min(history, key=lambda m: m.val_loss if m.val_loss is not None else m.train_loss)
            print(f"\n  Best epoch: {best.epoch}  |  loss: {best.val_loss or best.train_loss:.4f}")
        print("  Training complete.\n")

    def show_test_results(self, results: dict) -> None:
        print("\n  Test Results:")
        for k, v in results.items():
            print(f"    {k}: {v:.4f}")
        print()

    def show_message(self, msg: str) -> None:
        print(f"  {msg}")
