"""
Full Grassmann Transcription Model.

Audio → Mel Spectrogram features → Grassmann Layers → CTC Output

Architecture:
  1. Conv frontend: projects mel features into d_model
  2. Positional encoding
  3. N x CausalGrassmannLayer
  4. Linear CTC head → vocab_size
"""

import math

import torch
import torch.nn as nn

from .config import GrassmannConfig
from .grassmann_layer import CausalGrassmannLayer


class ConvFrontend(nn.Module):
    """Convolutional feature extractor for mel spectrograms."""

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, time) - mel spectrogram

        Returns:
            (batch, time // 4, d_model) - downsampled and projected
        """
        out = self.conv(x)  # (B, d_model, T')
        return out.transpose(1, 2)  # (B, T', d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GrassmannTranscriptionModel(nn.Module):
    """
    Full model: Audio features → Grassmann layers → CTC output.

    Uses CTC (Connectionist Temporal Classification) loss for
    alignment-free training on audio transcription.
    """

    def __init__(self, config: GrassmannConfig):
        super().__init__()
        self.config = config

        # Frontend: mel → d_model
        self.frontend = ConvFrontend(config.n_mels, config.d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        # Grassmann layers
        self.layers = nn.ModuleList(
            [CausalGrassmannLayer(config, i) for i in range(config.n_layers)]
        )

        # CTC output head
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time) - mel spectrogram

        Returns:
            log_probs: (time', batch, vocab_size) - CTC-formatted log probs
        """
        # Conv frontend
        x = self.frontend(mel)  # (B, T', d_model)

        # Positional encoding
        x = self.pos_enc(x)

        # Grassmann layers
        for layer in self.layers:
            x = layer(x)

        # CTC head
        logits = self.output_proj(x)  # (B, T', vocab_size)
        log_probs = logits.log_softmax(dim=-1)

        # CTC expects (T, B, C)
        return log_probs.transpose(0, 1)

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate output sequence lengths after conv downsampling."""
        # Two conv layers with stride 2 each → /4
        lengths = input_lengths
        for _ in range(2):
            lengths = (lengths + 1) // 2
        return lengths
