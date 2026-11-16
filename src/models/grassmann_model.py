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
import torch.nn.functional as F

from .config import GrassmannConfig
from .grassmann_layer import CausalGrassmannLayer
from .ablation_layer import AblationLayer
from .conformer_block import GrassmannConformerBlock, AblationConformerBlock


class ConvFrontend(nn.Module):
    """Conv1d feature extractor for mel spectrograms (legacy)."""

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)  # (B, d_model, T')
        return out.transpose(1, 2)  # (B, T', d_model)


class Conv2dSubsampling(nn.Module):
    """
    Conv2d subsampling from Conformer (Gulati et al. 2020).

    Two 3x3 stride-2 convolutions: 10ms fbank frames → 40ms encoder frames.
    Standard ASR frontend used by ESPnet, WeNet, etc.
    """

    def __init__(self, n_mels: int, channels: int, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2)
        # After two 3x3 stride-2 convs: freq dim = ((n_mels - 1) // 2 - 1) // 2
        freq_out = ((n_mels - 1) // 2 - 1) // 2
        self.out = nn.Linear(channels * freq_out, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, time) - mel spectrogram

        Returns:
            (batch, time', d_model)
        """
        # Reshape: (B, n_mels, T) → (B, 1, T, n_mels) for Conv2d
        x = x.transpose(1, 2).unsqueeze(1)  # (B, 1, T, n_mels)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        b, c, t, f = x.shape
        x = x.transpose(1, 2).reshape(b, t, c * f)  # (B, T', C*F)
        return self.out(x)  # (B, T', d_model)


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

    Supports two encoder modes:
      - Plain: CausalGrassmannLayer stack (original)
      - Conformer: GrassmannConformerBlock stack (with ConvModule + Macaron FFN)

    And two frontend modes:
      - Conv1d (legacy)
      - Conv2d subsampling (proper ASR, from Conformer)
    """

    def __init__(self, config: GrassmannConfig):
        super().__init__()
        self.config = config

        # Frontend: mel → d_model
        if config.use_conformer:
            self.frontend = Conv2dSubsampling(
                config.n_mels, config.conv_channels, config.d_model
            )
        else:
            self.frontend = ConvFrontend(config.n_mels, config.d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        # Encoder layers
        ablation = getattr(config, "use_ablation", False)
        if config.use_conformer:
            block_cls = AblationConformerBlock if ablation else GrassmannConformerBlock
            self.layers = nn.ModuleList(
                [block_cls(config) for _ in range(config.n_layers)]
            )
        else:
            layer_cls = AblationLayer if ablation else CausalGrassmannLayer
            self.layers = nn.ModuleList(
                [layer_cls(config, i) for i in range(config.n_layers)]
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
        lengths = input_lengths
        if self.config.use_conformer:
            # Conv2d subsampling: two 3x3 stride-2 convs (no padding)
            # Each conv: out = (in - 3) // 2 + 1 = (in - 1) // 2
            for _ in range(2):
                lengths = (lengths - 1) // 2
        else:
            # Conv1d frontend: two stride-2 convs with padding=1
            for _ in range(2):
                lengths = (lengths + 1) // 2
        return lengths
