"""
Grassmann Cross-Mixing Layer.

Replaces cross-attention in the decoder with Grassmann geometry.
Instead of Q*K^T dot products between decoder and encoder states,
we form Plücker coordinates between decoder positions and encoder
positions within a local monotonic window.

This is the key novel component — cross-sequence interaction via
subspace geometry on Gr(2, r) rather than attention weights.
"""

import torch
import torch.nn as nn

from .grassmann_layer import PluckerEncoder
from .config import GrassmannConfig


class GrassmannCrossLayer(nn.Module):
    """
    Cross-mixing between decoder and encoder via Grassmann/Plücker geometry.

    For each decoder position t, estimates an alignment position in the
    encoder sequence, then computes Plücker features between the decoder
    state and encoder states in a window around that alignment.

    This replaces cross-attention entirely — no Q*K^T, no softmax,
    just subspace geometry on a manifold.
    """

    def __init__(self, config: GrassmannConfig):
        super().__init__()
        d = config.d_model
        r = config.r_reduced
        plucker_dim = config.plucker_dim

        # Separate reduction projections for decoder and encoder
        self.W_red_dec = nn.Linear(d, r)
        self.W_red_enc = nn.Linear(d, r)

        # Plücker encoder (shared geometry)
        self.plucker = PluckerEncoder(r, eps=config.plucker_eps)

        # Project Plücker features back to model dim
        self.W_plu = nn.Linear(plucker_dim, d)

        # Gated fusion: blend decoder state with cross-mixed features
        self.W_gate = nn.Linear(2 * d, d)

        # Layer norm
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

        # Cross-mixing window: how many encoder positions to look at
        # per decoder position (on each side of the alignment point)
        self.cross_window = config.cross_window

    def _estimate_alignment(
        self, dec_len: int, enc_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Estimate monotonic alignment: decoder position t maps to
        encoder position t * (enc_len / dec_len).

        Returns integer encoder indices for each decoder position.
        """
        # Linear interpolation: dec pos 0..T_dec maps to enc pos 0..T_enc
        dec_positions = torch.arange(dec_len, device=device, dtype=torch.float)
        scale = enc_len / max(dec_len, 1)
        enc_positions = (dec_positions * scale).long().clamp(0, enc_len - 1)
        return enc_positions

    def forward(
        self, h_dec: torch.Tensor, h_enc: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_dec: (batch, T_dec, d_model) - decoder hidden states
            h_enc: (batch, T_enc, d_model) - encoder hidden states

        Returns:
            h_out: (batch, T_dec, d_model) - cross-mixed decoder states
        """
        B, T_dec, D = h_dec.shape
        T_enc = h_enc.shape[1]

        # Reduce both to Grassmann space
        z_dec = self.W_red_dec(h_dec)   # (B, T_dec, r)
        z_enc = self.W_red_enc(h_enc)   # (B, T_enc, r)

        # Estimate alignment positions
        align = self._estimate_alignment(T_dec, T_enc, h_dec.device)  # (T_dec,)

        # For each decoder position, gather encoder states in a window
        # around the alignment point and compute Plücker features
        w = self.cross_window
        g_accum = torch.zeros_like(h_dec)  # (B, T_dec, D)
        count = 0

        for offset in range(-w, w + 1):
            # Encoder indices for this offset
            enc_idx = (align + offset).clamp(0, T_enc - 1)  # (T_dec,)

            # Gather encoder states at these positions
            enc_gathered = z_enc[:, enc_idx, :]  # (B, T_dec, r)

            # Plücker encoding between decoder and encoder states
            p = self.plucker(z_dec, enc_gathered)  # (B, T_dec, plucker_dim)

            # Project back
            g = self.W_plu(p)  # (B, T_dec, D)
            g_accum += g
            count += 1

        g_cross = g_accum / max(count, 1)

        # Gated fusion
        u = torch.cat([h_dec, g_cross], dim=-1)  # (B, T_dec, 2D)
        alpha = torch.sigmoid(self.W_gate(u))      # (B, T_dec, D)
        h_mixed = alpha * h_dec + (1 - alpha) * g_cross

        # Residual + layer norm + dropout
        h_out = self.norm(h_dec + self.dropout(h_mixed))

        return h_out
