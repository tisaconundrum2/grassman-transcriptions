from abc import ABC, abstractmethod
from typing import Self

from torch.utils.data import DataLoader


class IDataset(ABC):
    """Interface for dataset loading and preprocessing."""

    @abstractmethod
    def load(self, path: str) -> Self:
        """Load dataset from path."""
        ...

    @abstractmethod
    def clean(self) -> Self:
        """Clean and filter the dataset."""
        ...

    @abstractmethod
    def preprocess(self) -> Self:
        """Preprocess dataset (extract features, tokenize labels)."""
        ...

    @abstractmethod
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Self:
        """Split into train/val/test sets."""
        ...

    @abstractmethod
    def get_train_loader(self, batch_size: int) -> DataLoader:
        ...

    @abstractmethod
    def get_val_loader(self, batch_size: int) -> DataLoader:
        ...

    @abstractmethod
    def get_test_loader(self, batch_size: int) -> DataLoader:
        ...
