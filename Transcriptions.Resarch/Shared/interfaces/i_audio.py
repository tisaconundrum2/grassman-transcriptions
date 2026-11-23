from abc import ABC, abstractmethod
from typing import Any

import torch


class IAudioProcessor(ABC):
    """Interface for audio feature extraction."""

    @abstractmethod
    def load(self, path: str) -> torch.Tensor:
        """Load raw audio from file. Returns waveform tensor."""
        ...

    @abstractmethod
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features (e.g. mel spectrogram) from waveform."""
        ...

    @abstractmethod
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features for model input."""
        ...
