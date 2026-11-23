from abc import ABC, abstractmethod

import torch


class ITokenizer(ABC):
    """Interface for text tokenization (transcription labels)."""

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Encode text string to token indices."""
        ...

    @abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token indices back to text string."""
        ...

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def blank_id(self) -> int:
        """CTC blank token id."""
        ...
