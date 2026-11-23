"""
Grassmann Decoder.

A fully attention-free autoregressive decoder using:
  1. Causal Grassmann self-mixing (same as encoder)
  2. Grassmann cross-mixing (replaces cross-attention)
  3. Position-wise FFN

Each decoder block: CausalSelfMix → CrossMix → FFN
Stacked M times, with token embeddings and output projection.
"""

import math

import torch
import torch.nn as nn

from .config import GrassmannConfig
from .grassmann_layer import CausalGrassmannLayer
from .grassmann_cross_layer import GrassmannCrossLayer


class GrassmannDecoderBlock(nn.Module):
    """
    One decoder block: self-mixing → cross-mixing → FFN.

    The self-mixing uses CausalGrassmannLayer (causal, decoder-to-decoder).
    The cross-mixing uses GrassmannCrossLayer (decoder-to-encoder).
    The FFN is already included in CausalGrassmannLayer, so we only add
    the cross-mixing step between self-mixing and the FFN residual.
    """

    def __init__(self, config: GrassmannConfig, layer_idx: int = 0):
        super().__init__()
        d = config.d_model

        # Causal self-mixing (includes FFN)
        self.self_mix = CausalGrassmannLayer(config, layer_idx)

        # Grassmann cross-mixing with encoder
        self.cross_mix = GrassmannCrossLayer(config)

        # Additional FFN after cross-mixing
        self.ffn = nn.Sequential(
            nn.Linear(d, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, d),
            nn.Dropout(config.dropout),
        )
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, h_dec: torch.Tensor, h_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_dec: (batch, T_dec, d_model) - decoder states
            h_enc: (batch, T_enc, d_model) - encoder output

        Returns:
            (batch, T_dec, d_model) - updated decoder states
        """
        # Self-mixing (causal, within decoder sequence)
        h = self.self_mix(h_dec)

        # Cross-mixing (decoder ↔ encoder via Grassmann geometry)
        h = self.cross_mix(h, h_enc)

        # FFN with residual
        h_out = self.norm_ffn(h + self.ffn(h))

        return h_out


class GrassmannDecoder(nn.Module):
    """
    Full Grassmann decoder: token embeddings → M decoder blocks → output logits.

    Fully attention-free. All sequence interactions happen through
    Plücker coordinates on Gr(2, r).
    """

    def __init__(self, config: GrassmannConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.embed_scale = math.sqrt(config.d_model)

        # Decoder blocks
        self.blocks = nn.ModuleList([
            GrassmannDecoderBlock(config, i)
            for i in range(config.n_decoder_layers)
        ])

        # Output projection to vocab
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        encoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens:      (batch, T_dec) - input token indices (shifted right)
            encoder_out: (batch, T_enc, d_model) - encoder hidden states

        Returns:
            logits: (batch, T_dec, vocab_size)
        """
        B, T = tokens.shape

        # Token + positional embeddings
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        h = self.token_embed(tokens) * self.embed_scale + self.pos_embed(positions)
        h = self.embed_dropout(h)

        # Decoder blocks
        for block in self.blocks:
            h = block(h, encoder_out)

        # Output
        h = self.output_norm(h)
        logits = self.output_proj(h)

        return logits
