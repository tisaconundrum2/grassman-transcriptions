"""
Causal Grassmann Mixing Layer.

Implements the core architecture from "Attention Is Not What You Need"
(arXiv:2512.19428). Replaces self-attention with:

1. Linear reduction:       h_t ∈ R^d  →  z_t ∈ R^r
2. Multi-scale pairing:    (z_t, z_{t+Δ}) for Δ ∈ W
3. Plücker encoding:       p_{ij} = z_{t,i} * z_{t+Δ,j} - z_{t,j} * z_{t+Δ,i}
4. Projection back:        g_t = W_plu @ p_hat + b_plu
5. Gated fusion:           h_mix = α ⊙ h + (1-α) ⊙ g
6. Feed-forward block:     standard FFN with GELU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GrassmannConfig


class PluckerEncoder(nn.Module):
    """Computes Plücker coordinates for pairs of reduced vectors."""

    def __init__(self, r: int, eps: float = 1e-8):
        super().__init__()
        self.r = r
        self.eps = eps
        # Precompute index pairs (i, j) where i < j
        idx_i, idx_j = [], []
        for i in range(r):
            for j in range(i + 1, r):
                idx_i.append(i)
                idx_j.append(j)
        self.register_buffer("idx_i", torch.tensor(idx_i, dtype=torch.long))
        self.register_buffer("idx_j", torch.tensor(idx_j, dtype=torch.long))

    def forward(self, z_t: torch.Tensor, z_delta: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized Plücker coordinates.

        Args:
            z_t:     (batch, seq, r) - reduced states at position t
            z_delta: (batch, seq, r) - reduced states at position t+Δ

        Returns:
            (batch, seq, C(r,2)) - normalized Plücker vectors
        """
        # p_{ij} = z_{t,i} * z_{Δ,j} - z_{t,j} * z_{Δ,i}
        p = (
            z_t[..., self.idx_i] * z_delta[..., self.idx_j]
            - z_t[..., self.idx_j] * z_delta[..., self.idx_i]
        )
        # Normalize for numerical stability
        norm = p.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return p / norm


class CausalGrassmannLayer(nn.Module):
    """
    One layer of the Causal Grassmann Transformer.

    Replaces the self-attention block with Grassmann/Plücker mixing,
    followed by the standard position-wise FFN.
    """

    def __init__(self, config: GrassmannConfig, layer_idx: int = 0):
        super().__init__()
        d = config.d_model
        r = config.r_reduced
        plucker_dim = config.plucker_dim

        # Select window offset for this layer (cycle through offsets)
        self.window_offsets = config.window_offsets

        # Step 1: Linear reduction  h ∈ R^d → z ∈ R^r
        self.W_red = nn.Linear(d, r)

        # Step 3: Plücker encoder
        self.plucker = PluckerEncoder(r, eps=config.plucker_eps)

        # Step 4: Project Plücker features back to model dim
        self.W_plu = nn.Linear(plucker_dim, d)

        # Step 5: Gated fusion
        self.W_gate = nn.Linear(2 * d, d)

        # Layer norm after mixing
        self.norm_mix = nn.LayerNorm(d)
        self.dropout_mix = nn.Dropout(config.dropout)

        # Step 6: Feed-forward block
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, d),
            nn.Dropout(config.dropout),
        )
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, seq_len, d_model)

        Returns:
            h': (batch, seq_len, d_model)
        """
        B, L, D = h.shape

        # Step 1: Linear reduction
        z = self.W_red(h)  # (B, L, r)

        # Steps 2-4: Multi-scale Plücker features, aggregated
        g_accum = torch.zeros_like(h)  # (B, L, D)
        count = torch.zeros(B, L, 1, device=h.device)

        for delta in self.window_offsets:
            if delta >= L:
                continue
            # Causal pairing: pair t with t+Δ (looking ahead for encoding,
            # but we assign the feature to position t+Δ to maintain causality -
            # position t+Δ can see position t which is in its past)
            z_t = z[:, :L - delta, :]       # (B, L-Δ, r)
            z_d = z[:, delta:, :]            # (B, L-Δ, r)

            # Plücker encoding
            p = self.plucker(z_t, z_d)       # (B, L-Δ, plucker_dim)

            # Project back to model space
            g_delta = self.W_plu(p)          # (B, L-Δ, D)

            # Accumulate: assign to the later position (causal)
            g_accum[:, delta:, :] += g_delta
            count[:, delta:, :] += 1

        # Average over valid offsets
        count = count.clamp(min=1)
        g = g_accum / count  # (B, L, D)

        # Step 5: Gated fusion
        u = torch.cat([h, g], dim=-1)       # (B, L, 2D)
        alpha = torch.sigmoid(self.W_gate(u))  # (B, L, D)
        h_mix = alpha * h + (1 - alpha) * g

        # Layer norm + dropout + residual
        h_hat = self.norm_mix(h + self.dropout_mix(h_mix))

        # Step 6: Feed-forward with residual
        h_out = self.norm_ffn(h_hat + self.ffn(h_hat))

        return h_out
