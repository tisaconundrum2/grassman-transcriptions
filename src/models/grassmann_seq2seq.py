"""
Grassmann Seq2Seq Model — Fully Attention-Free Encoder-Decoder.

Audio → Grassmann Encoder → Grassmann Decoder → Text

Every sequence interaction in this model happens through Plücker
coordinates on the Grassmann manifold Gr(2, r). There are zero
attention matrices anywhere in the architecture:

  Encoder self-mixing:  Plücker(z_t, z_{t+Δ})  — local causal windows
  Decoder self-mixing:  Plücker(z_t, z_{t+Δ})  — causal autoregressive
  Decoder cross-mixing: Plücker(z_dec, z_enc)   — monotonic alignment windows

Trained with teacher forcing and cross-entropy loss.
"""

import torch
import torch.nn as nn

from .config import GrassmannConfig
from .grassmann_model import ConvFrontend, PositionalEncoding
from .grassmann_layer import CausalGrassmannLayer
from .grassmann_decoder import GrassmannDecoder


class GrassmannEncoder(nn.Module):
    """Audio encoder: mel spectrogram → Grassmann layers → encoder states."""

    def __init__(self, config: GrassmannConfig):
        super().__init__()
        self.frontend = ConvFrontend(config.n_mels, config.d_model)
        self.pos_enc = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.layers = nn.ModuleList([
            CausalGrassmannLayer(config, i) for i in range(config.n_layers)
        ])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time) - mel spectrogram

        Returns:
            (batch, T_enc, d_model) - encoder hidden states
        """
        x = self.frontend(mel)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x


class GrassmannSeq2SeqModel(nn.Module):
    """
    Fully attention-free encoder-decoder for audio transcription.

    Architecture:
      Encoder: ConvFrontend → PosEnc → N × CausalGrassmannLayer
      Decoder: TokenEmbed → PosEmbed → M × (CausalGrassmann + CrossGrassmann + FFN)
      Output:  Linear → vocab logits

    All interactions use Plücker coordinates. Zero attention matrices.
    """

    # Special token IDs
    SOS_ID = 0  # Start of sequence (overloads blank/pad)
    EOS_ID = 0  # End of sequence (same as SOS for simplicity)
    PAD_ID = 0  # Padding

    def __init__(self, config: GrassmannConfig):
        super().__init__()
        self.config = config
        self.encoder = GrassmannEncoder(config)
        self.decoder = GrassmannDecoder(config)

    def forward(
        self,
        mel: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass.

        Args:
            mel:           (batch, n_mels, time) - input mel spectrogram
            target_tokens: (batch, T_target) - target token indices

        Returns:
            logits: (batch, T_target, vocab_size)
        """
        # Encode audio
        encoder_out = self.encoder(mel)  # (B, T_enc, d)

        # Decode with teacher forcing
        # Input to decoder is target shifted right (prepend SOS)
        dec_input = self._shift_right(target_tokens)
        logits = self.decoder(dec_input, encoder_out)

        return logits

    def _shift_right(self, tokens: torch.Tensor) -> torch.Tensor:
        """Shift tokens right and prepend SOS token for teacher forcing."""
        sos = torch.full(
            (tokens.shape[0], 1), self.SOS_ID,
            dtype=tokens.dtype, device=tokens.device
        )
        # Prepend SOS, drop last token
        return torch.cat([sos, tokens[:, :-1]], dim=1)

    def get_encoder_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate encoder output lengths after conv downsampling."""
        lengths = input_lengths
        for _ in range(2):  # 2 conv layers with stride 2
            lengths = (lengths + 1) // 2
        return lengths

    @torch.no_grad()
    def greedy_decode(
        self, mel: torch.Tensor, max_len: int = 200, eos_id: int = 0
    ) -> torch.Tensor:
        """
        Autoregressive greedy decoding.

        Args:
            mel:     (1, n_mels, time) - single mel spectrogram
            max_len: maximum output length
            eos_id:  end-of-sequence token ID

        Returns:
            (seq_len,) - predicted token indices
        """
        self.eval()
        device = mel.device

        # Encode once
        encoder_out = self.encoder(mel)  # (1, T_enc, d)

        # Start with SOS token
        generated = [self.SOS_ID]

        for _ in range(max_len):
            # Build decoder input from generated tokens so far
            dec_input = torch.tensor(
                [generated], dtype=torch.long, device=device
            )  # (1, t)

            # Decode
            logits = self.decoder(dec_input, encoder_out)  # (1, t, vocab)

            # Take the last position's prediction
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)

            if next_token == eos_id:
                break

        # Return without the initial SOS
        return torch.tensor(generated[1:], dtype=torch.long)
