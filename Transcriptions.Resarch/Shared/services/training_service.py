from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Shared.models.config import GrassmannConfig


@dataclass
class TrainMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None = None


class TrainingService:
    """Handles the training loop, loss computation, and optimization."""

    def __init__(
        self,
        model: nn.Module,
        config: GrassmannConfig,
        device: torch.device | None = None,
    ):
        self.model = model
        self.config = config
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # MPS can be unstable with some PyTorch/Python combos; prefer CPU as fallback
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.history: list[TrainMetrics] = []

        # AMP: mixed precision for faster training on GPU
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def set_cosine_schedule(self, epochs: int, warmup_epochs: int = 2) -> None:
        """Add cosine annealing LR schedule with linear warmup."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_epochs])

    def state_dict(self) -> dict:
        """Optimizer, LR scheduler, and AMP scaler state for exact resume."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.use_amp and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])

    def fast_forward_schedule(self, epochs_done: int) -> None:
        """Advance the LR schedule as if `epochs_done` epochs had run.

        Used when resuming from a checkpoint that has no saved scheduler state.
        """
        if self.scheduler is None:
            return
        import warnings
        with warnings.catch_warnings():
            # Stepping before optimizer.step() warns; harmless here.
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(epochs_done):
                self.scheduler.step()

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        total_batches = len(train_loader)

        for batch_idx, (mels, tokens, mel_lengths, token_lengths) in enumerate(train_loader):
            mels = mels.to(self.device)
            tokens = tokens.to(self.device)
            mel_lengths = mel_lengths.to(self.device)
            token_lengths = token_lengths.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                log_probs = self.model(mels)  # (T', B, vocab)
                output_lengths = self.model.get_output_lengths(mel_lengths)
                loss = self.ctc_loss(log_probs, tokens, output_lengths, token_lengths)

            if torch.isfinite(loss):
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
                n_batches += 1

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg = total_loss / max(n_batches, 1)
                print(f"    batch {batch_idx+1}/{total_batches}  loss: {avg:.4f}", flush=True)

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for mels, tokens, mel_lengths, token_lengths in val_loader:
            mels = mels.to(self.device)
            tokens = tokens.to(self.device)
            mel_lengths = mel_lengths.to(self.device)
            token_lengths = token_lengths.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                log_probs = self.model(mels)
                output_lengths = self.model.get_output_lengths(mel_lengths)
                loss = self.ctc_loss(log_probs, tokens, output_lengths, token_lengths)

            if torch.isfinite(loss):
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        callback: object | None = None,
    ) -> list[TrainMetrics]:
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader) if val_loader else None

            metrics = TrainMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            self.history.append(metrics)

            if callback and hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(metrics)

            if self.scheduler is not None:
                self.scheduler.step()

        return self.history

    @torch.no_grad()
    def greedy_decode(self, mel: torch.Tensor) -> torch.Tensor:
        """Greedy CTC decoding for a single mel spectrogram."""
        self.model.eval()
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # add batch dim
        mel = mel.to(self.device)
        log_probs = self.model(mel)  # (T', 1, vocab)
        pred = log_probs.argmax(dim=-1).squeeze(1)  # (T',)
        return pred
