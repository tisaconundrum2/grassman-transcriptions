"""
Seq2Seq Training Service.

Handles teacher-forced training for the Grassmann encoder-decoder model.
Uses cross-entropy loss instead of CTC, with proper token shifting
and padding masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.config import GrassmannConfig
from src.models.grassmann_seq2seq import GrassmannSeq2SeqModel


@dataclass
class TrainMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None = None


class Seq2SeqTrainingService:
    """Training loop for the fully attention-free encoder-decoder."""

    def __init__(
        self,
        model: GrassmannSeq2SeqModel,
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
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        # Cross-entropy loss, ignoring padding (index 0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.history: list[TrainMetrics] = []

    def set_cosine_schedule(self, epochs: int, warmup_epochs: int = 2) -> None:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_epochs])

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for mels, tokens, mel_lengths, token_lengths in train_loader:
            mels = mels.to(self.device)
            tokens = tokens.to(self.device)

            self.optimizer.zero_grad()

            # Teacher-forced forward pass
            logits = self.model(mels, tokens)  # (B, T_dec, vocab)

            # Reshape for cross-entropy: (B*T, vocab) vs (B*T,)
            B, T, V = logits.shape
            loss = self.criterion(logits.reshape(B * T, V), tokens.reshape(B * T))

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for mels, tokens, mel_lengths, token_lengths in val_loader:
            mels = mels.to(self.device)
            tokens = tokens.to(self.device)

            logits = self.model(mels, tokens)

            B, T, V = logits.shape
            loss = self.criterion(logits.reshape(B * T, V), tokens.reshape(B * T))

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
        """Autoregressive greedy decoding for a single mel spectrogram."""
        self.model.eval()
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.to(self.device)
        return self.model.greedy_decode(mel, max_len=200)
