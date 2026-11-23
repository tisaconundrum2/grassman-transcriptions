from abc import ABC, abstractmethod
from typing import Iterator

import torch

Sample = tuple[torch.Tensor, torch.Tensor]


class IShardRepository(ABC):
    """Interface for fetching pre-cached mel shards.

    Implementations decide *how* shards arrive (mounted volume, Hub download,
    Hub download with prefetch). Consumers only see lists of samples.
    """

    @abstractmethod
    def list_shards(self) -> list[str]:
        """Sorted shard file names (e.g. 'shard_0000.pt')."""
        ...

    @abstractmethod
    def total_samples(self) -> int:
        """Total samples across all shards (from metadata, or an estimate)."""
        ...

    @abstractmethod
    def iter_shards(self, shard_names: list[str]) -> Iterator[list[Sample]]:
        """Yield the loaded contents of each shard, in the given order.

        Only one shard's samples should be held in RAM at a time.
        """
        ...
