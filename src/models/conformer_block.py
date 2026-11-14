"""
Conformer Block with Grassmann Mixer.

Macaron-style block from Gulati et al. 2020 (arXiv 2005.08100)
with the self-attention slot replaced by a Grassmann mixer:

  x → +0.5*FFN → +Mixer → +Conv → +0.5*FFN → LayerNorm

This gives the best of both worlds:
  - Grassmann mixer for sequence-level subspace geometry
  - ConvModule for local temporal pattern capture
  - Macaron FFN sandwich for nonlinear capacity
"""

import torch
import torch.nn as nn

from .config import GrassmannConfig
from .conv_module import ConvModule
from .grassmann_layer import GrassmannMixer


def _macaron_ffn(d: int, d_ff: int, dropout: float) -> nn.Sequential:
    """Conformer-style FFN: LayerNorm → Linear → SiLU → Dropout → Linear → Dropout."""
    return nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, d_ff),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d),
        nn.Dropout(dropout),
    )


class GatedGrassmannMixer(nn.Module):
    """
    Two-sided Grassmann mix with sigmoid gate as a residual branch.
    (1 - α)(g - u) formulation from Dhruv's implementation.
    """

    def __init__(self, d: int, r: int, offsets: list[int], eps: float = 1e-8):
        super().__init__()
        self.mixer = GrassmannMixer(d, r, offsets, two_sided=True, eps=eps)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        g = self.mixer(u)
        alpha = torch.sigmoid(self.gate(torch.cat([u, g], dim=-1)))
        return (1 - alpha) * (g - u)


class GrassmannConformerBlock(nn.Module):
    """
    Conformer block with Grassmann mixer replacing self-attention.

    Architecture (Macaron block):
      x → +0.5*FFN1 → +GatedGrassmannMixer → +ConvModule → +0.5*FFN2 → LayerNorm
    """

    def __init__(self, config: GrassmannConfig):
        super().__init__()
        d = config.d_model
        d_ff = config.d_ff
        r = config.r_reduced
        dropout = config.dropout
        kernel = getattr(config, "conv_kernel_size", 31)
        offsets = config.window_offsets

        # Macaron FFN sandwich
        self.ff1 = _macaron_ffn(d, d_ff, dropout)
        self.ff2 = _macaron_ffn(d, d_ff, dropout)

        # Grassmann mixer (replaces self-attention)
        self.ln_mix = nn.LayerNorm(d)
        self.mixer = GatedGrassmannMixer(d, r, offsets, eps=config.plucker_eps)
        self.drop = nn.Dropout(dropout)

        # ConvModule for local patterns
        self.conv = ConvModule(d, kernel, dropout)

        # Final LayerNorm
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Half-step FFN 1
        x = x + 0.5 * self.ff1(x)
        # Grassmann mixer (with gate as residual)
        x = x + self.drop(self.mixer(self.ln_mix(x)))
        # ConvModule
        x = x + self.conv(x)
        # Half-step FFN 2 + final norm
        return self.ln_out(x + 0.5 * self.ff2(x))
