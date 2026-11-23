"""
Conformer ConvModule.

Captures local temporal patterns that Grassmann mixing misses.
From Gulati et al. 2020 (arXiv 2005.08100):

  LayerNorm → Pointwise Conv (GLU) → Depthwise Conv → BatchNorm → SiLU → Pointwise Conv → Dropout

This is the key component that makes Conformer superior to pure
Transformer/Grassmann for speech — it captures local phonetic patterns
within a convolutional receptive field.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Conformer convolution module with depthwise separable convolution."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        # Pointwise expansion with GLU gating
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        # Depthwise convolution (each channel independently)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.bn = nn.BatchNorm1d(d_model)
        # Pointwise projection back
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # LayerNorm → transpose for Conv1d
        y = self.ln(x).transpose(1, 2)  # (B, d, T)
        # Pointwise expansion + GLU gate
        y = F.glu(self.pw1(y), dim=1)   # (B, d, T)
        # Depthwise conv + BatchNorm + SiLU
        y = F.silu(self.bn(self.dw(y))) # (B, d, T)
        # Pointwise projection + dropout
        y = self.drop(self.pw2(y))       # (B, d, T)
        return y.transpose(1, 2)         # (B, T, d)
