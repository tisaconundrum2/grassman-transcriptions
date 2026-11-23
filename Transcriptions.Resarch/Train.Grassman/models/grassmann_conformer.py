"""
Grassmann Conformer Block.

Macaron-style block from Gulati et al. 2020 (arXiv 2005.08100)
with the self-attention slot replaced by a Grassmann mixer:

  x → +0.5*FFN → +Mixer → +Conv → +0.5*FFN → LayerNorm
"""

import torch
import torch.nn as nn

from Shared.models.config import GrassmannConfig
from Shared.models.conv_module import ConvModule
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

        self.ff1 = _macaron_ffn(d, d_ff, dropout)
        self.ff2 = _macaron_ffn(d, d_ff, dropout)

        self.ln_mix = nn.LayerNorm(d)
        self.mixer = GatedGrassmannMixer(d, r, offsets, eps=config.plucker_eps)
        self.drop = nn.Dropout(dropout)

        self.conv = ConvModule(d, kernel, dropout)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.mixer(self.ln_mix(x)))
        x = x + self.conv(x)
        return self.ln_out(x + 0.5 * self.ff2(x))
