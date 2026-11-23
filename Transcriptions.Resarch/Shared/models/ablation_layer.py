"""
Ablation Mixing Layer — Symmetric Product Baseline.

Drop-in replacement for the Grassmann/Plücker mixer that uses plain
element-wise symmetric products instead of antisymmetric Plücker
coordinates.  Everything else (linear reduction, multi-scale offsets,
mean-first anchoring, gated fusion, FFN) stays identical so the only
variable is whether the Grassmann geometry matters.

Matches Dhruv's grassmann-speech implementation:
  - symmetric_products:  a_i * b_j + a_j * b_i  (vs Plücker: a_i * b_j - a_j * b_i)
  - symmetric_norm_sq:   |a|²|b|² + (a·b)² - 2·Σ(aᵢ²bᵢ²)  (vs Lagrange: |a|²|b|² - (a·b)²)
  - flip = +1.0 in two-sided mode  (vs Plücker: flip = -1.0)
"""

import torch
import torch.nn as nn

from .config import GrassmannConfig


class SymmetricEncoder(nn.Module):
    """
    Computes element-wise symmetric products for pairs of reduced vectors.

    Same index pairs and output dimensionality as PluckerEncoder,
    but uses  p_{ij} = z_{t,i} * z_{Δ,j} + z_{t,j} * z_{Δ,i}
    instead of the antisymmetric Plücker form.
    """

    def __init__(self, r: int, eps: float = 1e-8):
        super().__init__()
        self.r = r
        self.eps = eps
        idx_i, idx_j = [], []
        for i in range(r):
            for j in range(i + 1, r):
                idx_i.append(i)
                idx_j.append(j)
        self.register_buffer("idx_i", torch.tensor(idx_i, dtype=torch.long))
        self.register_buffer("idx_j", torch.tensor(idx_j, dtype=torch.long))

    def forward(self, z_t: torch.Tensor, z_delta: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized symmetric products.

        Args:
            z_t:     (batch, seq, r)
            z_delta: (batch, seq, r)

        Returns:
            (batch, seq, C(r,2))
        """
        # Symmetric: p_{ij} = z_{t,i} * z_{Δ,j} + z_{t,j} * z_{Δ,i}
        p = (
            z_t[..., self.idx_i] * z_delta[..., self.idx_j]
            + z_t[..., self.idx_j] * z_delta[..., self.idx_i]
        )
        norm = p.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return p / norm


def _symmetric_norm_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Squared norm of symmetric products from dot products alone.
    Matches grassmann-speech/plucker.py symmetric_norm_sq:
      |a|²|b|² + (a·b)² - 2·Σ(aᵢ²·bᵢ²)
    """
    aa = (a * a).sum(-1, keepdim=True)
    bb = (b * b).sum(-1, keepdim=True)
    ab = (a * b).sum(-1, keepdim=True)
    return aa * bb + ab * ab - 2 * (a * a * b * b).sum(-1, keepdim=True)


class AblationMixer(nn.Module):
    """
    Symmetric-product mixer — ablation baseline for GrassmannMixer.

    Identical architecture (linear reduction → multi-scale pairing →
    encoding → mean-first anchoring → projection) but with symmetric
    products replacing Plücker coordinates.

    Key differences from GrassmannMixer (matching grassmann-speech):
      1. Symmetric products instead of Plücker coordinates
      2. symmetric_norm_sq for anchored weighting
      3. flip = +1.0 in two-sided mode (symmetric products are symmetric
         under swap, so no sign flip — use positive accumulation)
    """

    def __init__(self, d: int, r: int, offsets: list[int],
                 two_sided: bool = False, eps: float = 1e-8):
        super().__init__()
        self.offsets = offsets
        self.two_sided = two_sided
        self.eps = eps
        plucker_dim = r * (r - 1) // 2  # same dimensionality

        self.W_red = nn.Linear(d, r)
        self.encoder = SymmetricEncoder(r, eps=eps)
        self.W_plu = nn.Linear(plucker_dim, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Mean-first (anchored) form, matching grassmann-speech."""
        B, L, D = h.shape
        z = self.W_red(h)

        partner = torch.zeros_like(z)
        count = z.new_zeros(1, L, 1)

        for delta in self.offsets:
            if delta >= L:
                continue
            earlier = z[:, :L - delta, :]
            later = z[:, delta:, :]

            # Weight by symmetric norm (from dot products, no coordinate expansion)
            w = _symmetric_norm_sq(earlier, later).clamp_min(0).sqrt().clamp_min(self.eps).reciprocal()

            partner[:, delta:] += w * earlier
            count[:, delta:] += 1

            if self.two_sided:
                # flip = +1.0 for symmetric (vs -1.0 for Plücker)
                # Symmetric products are invariant under swap(a,b),
                # so no sign flip needed to distinguish direction
                partner[:, :L - delta] += w * later
                count[:, :L - delta] += 1

        count = count.clamp(min=1)
        avg_partner = partner / count
        p_final = self.encoder(z, avg_partner)
        return self.W_plu(p_final)


class AblationLayer(nn.Module):
    """
    Drop-in replacement for CausalGrassmannLayer using symmetric products.

    Same gated fusion + FFN, just swaps the mixer.
    """

    def __init__(self, config: GrassmannConfig, layer_idx: int = 0):
        super().__init__()
        d = config.d_model
        r = config.r_reduced
        two_sided = getattr(config, "two_sided", False)

        self.mixer = AblationMixer(
            d, r, config.window_offsets,
            two_sided=two_sided, eps=config.plucker_eps,
        )

        self.W_gate = nn.Linear(2 * d, d)
        self.norm_mix = nn.LayerNorm(d)
        self.dropout_mix = nn.Dropout(config.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, d),
            nn.Dropout(config.dropout),
        )
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g = self.mixer(h)

        u = torch.cat([h, g], dim=-1)
        alpha = torch.sigmoid(self.W_gate(u))
        h_mix = h + (1 - alpha) * (g - h)

        h_hat = self.norm_mix(h + self.dropout_mix(h_mix))
        h_out = self.norm_ffn(h_hat + self.ffn(h_hat))
        return h_out
