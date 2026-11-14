from dataclasses import dataclass, field


@dataclass
class GrassmannConfig:
    """Configuration for the Grassmann Transcription Model."""

    # Audio features
    n_mels: int = 80
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160

    # Model architecture
    d_model: int = 256
    r_reduced: int = 32
    n_layers: int = 6
    d_ff: int = 1024  # 4 * d_model
    dropout: float = 0.1

    # Grassmann-specific
    window_offsets: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 12, 16])
    plucker_eps: float = 1e-8

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    epochs: int = 30
    max_seq_len: int = 512

    # Vocab (set dynamically by tokenizer)
    vocab_size: int = 29  # 26 letters + space + apostrophe + blank

    @property
    def plucker_dim(self) -> int:
        """Dimension of Plücker coordinate vector: C(r, 2)."""
        return self.r_reduced * (self.r_reduced - 1) // 2
