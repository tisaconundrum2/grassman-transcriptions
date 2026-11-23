"""
Ablation Conformer Block — Symmetric Product Baseline.

Macaron-style Conformer block with symmetric-product mixer
replacing self-attention. Drop-in replacement for GrassmannConformerBlock.
"""

import torch
import torch.nn as nn

from Shared.models.config import GrassmannConfig
from Shared.models.conv_module import ConvModule
from .ablation_layer import AblationMixer


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


class GatedAblationMixer(nn.Module):
    """
    Ablation baseline: symmetric-product mixer with the same gate.
    Drop-in replacement for GatedGrassmannMixer.
    """

    def __init__(self, d: int, r: int, offsets: list[int], eps: float = 1e-8):
        super().__init__()
        self.mixer = AblationMixer(d, r, offsets, two_sided=True, eps=eps)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        g = self.mixer(u)
        alpha = torch.sigmoid(self.gate(torch.cat([u, g], dim=-1)))
        return (1 - alpha) * (g - u)


class AblationConformerBlock(nn.Module):
    """
    Conformer block with symmetric-product mixer (ablation baseline).
    Same Macaron FFN + ConvModule, just swaps Grassmann for symmetric products.
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
        self.mixer = GatedAblationMixer(d, r, offsets, eps=config.plucker_eps)
        self.drop = nn.Dropout(dropout)

        self.conv = ConvModule(d, kernel, dropout)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.mixer(self.ln_mix(x)))
        x = x + self.conv(x)
        return self.ln_out(x + 0.5 * self.ff2(x))
