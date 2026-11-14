from abc import ABC, abstractmethod
from typing import Any, Self

import torch


class IModel(ABC):
    """Interface for the trainable model."""

    @abstractmethod
    def build(self) -> Self:
        """Build / initialize the model architecture."""
        ...

    @abstractmethod
    def train(self, train_loader: Any, val_loader: Any, epochs: int) -> Self:
        """Train the model."""
        ...

    @abstractmethod
    def test(self, test_loader: Any) -> dict:
        """Evaluate on test set. Returns metrics dict."""
        ...

    @abstractmethod
    def fit(self, dataset: Any) -> Self:
        """Convenience: preprocess + train + validate in one call."""
        ...

    @abstractmethod
    def predict(self, audio: torch.Tensor) -> str:
        """Transcribe audio tensor to text."""
        ...

    @abstractmethod
    def save(self, path: str) -> Self:
        """Save model checkpoint."""
        ...

    @abstractmethod
    def load(self, path: str) -> Self:
        """Load model checkpoint."""
        ...
