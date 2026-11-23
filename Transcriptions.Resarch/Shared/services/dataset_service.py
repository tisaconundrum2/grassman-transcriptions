"""
Dataset Service — orchestrates data loading, splitting, and DataLoader creation.

Delegates external data access to repositories:
  - DatasetRepository: HF Hub fetching, schema resolution, streaming
  - CacheRepository: disk cache read/write, HF shard upload/download

Supports three data modes:
  - Pre-cached shards via an IShardRepository (local/mounted, Hub, prefetch)
  - Disk cache: raw datasets cached to .pt files (load_recipe)
  - Direct HF: stream or download from HuggingFace datasets
"""

from __future__ import annotations

import os
import random
from typing import Self

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from Shared.interfaces.i_shard_repository import IShardRepository
from Shared.repositories.cache_repository import CacheRepository
from Shared.repositories.dataset_repository import DatasetRepository
from Shared.repositories.shard_repository import make_shard_repository


# -- PyTorch Dataset wrappers ---------------------------------------------

class MaterializedAudioDataset(Dataset):
    """Pre-materialized dataset -- samples already in memory as (mel, tokens) pairs."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class DiskCachedDataset(Dataset):
    """Dataset backed by individual .pt files on disk (used by load_recipe)."""

    def __init__(self, cache: CacheRepository, indices: list[int]):
        self._cache = cache
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cache.load(self._indices[idx])


class HFAudioDataset(Dataset):
    """Wraps a (non-streaming) HuggingFace dataset split into a PyTorch Dataset."""

    def __init__(self, hf_dataset, repo: DatasetRepository,
                 audio_col: str, text_col: str):
        self.hf_dataset = hf_dataset
        self._repo = repo
        self.audio_col = audio_col
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.hf_dataset[idx]
        mel, text = self._repo.process_sample(sample, self.audio_col, self.text_col)
        tokens = self._repo.tokenizer.encode(text)
        return mel, tokens


class ShardStreamingDataset(IterableDataset):
    """Iterates samples from pre-cached mel shards.

    How shards are fetched (mounted volume, Hub download, prefetch) is
    delegated to an IShardRepository. This class only handles ordering,
    shuffling, and the max_samples cap.
    """

    def __init__(self, shard_repo: IShardRepository, shard_files: list[str],
                 total_samples: int, shuffle: bool = True,
                 max_samples: int | None = None):
        self._shard_repo = shard_repo
        self._shard_files = shard_files
        self._total = min(total_samples, max_samples) if max_samples else total_samples
        self._shuffle = shuffle
        self._max_samples = max_samples

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        shards = list(self._shard_files)
        if self._shuffle:
            random.shuffle(shards)

        yielded = 0
        shard_iter = self._shard_repo.iter_shards(shards)
        try:
            for shard_data in shard_iter:
                if self._shuffle:
                    random.shuffle(shard_data)
                for mel, tokens in shard_data:
                    if self._max_samples and yielded >= self._max_samples:
                        return
                    yield mel, tokens
                    yielded += 1
                del shard_data
        finally:
            shard_iter.close()


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad mels and tokens to batch-max lengths."""
    mels, tokens = zip(*batch)
    mel_lengths = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
    max_mel_len = mel_lengths.max().item()
    padded_mels = []
    for m in mels:
        pad_amount = max_mel_len - m.shape[-1]
        padded_mels.append(torch.nn.functional.pad(m, (0, pad_amount)))
    mel_batch = torch.stack(padded_mels)
    token_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    token_batch = pad_sequence(tokens, batch_first=True, padding_value=0)
    return mel_batch, token_batch, mel_lengths, token_lengths


# -- Service ---------------------------------------------------------------

class DatasetService:
    """Orchestrates data loading, splitting, and DataLoader creation.

    Usage:
        repo = DatasetRepository(config, tokenizer)
        cache = CacheRepository("checkpoints/mel_cache")
        svc = DatasetService(repo, cache)

        # Pre-cached shards (repo id or mounted path; fetch_mode picks the repository):
        svc.load_from_hub("tisaconundrum/grassmann-mel-cache")              # prefetch
        svc.load_from_hub("/data/mel-cache")                                # mounted volume
        svc.load_from_hub("tisaconundrum/grassmann-mel-cache", fetch_mode="stream")
        svc.load_from_hub("tisaconundrum/grassmann-mel-cache", max_samples=28539)
        svc.split()

        # Direct HF download:
        svc.load_dataset("openslr/librispeech_asr", split="train.clean.100")
        svc.split()

        train_loader = svc.get_train_loader(batch_size=32)
    """

    def __init__(self, repo: DatasetRepository, cache: CacheRepository):
        self._repo = repo
        self._cache = cache
        self._hf_train = None
        self._hf_val = None
        self._hf_test = None
        self._materialized: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._use_cache = False
        self._audio_col = "audio"
        self._text_col = "text"
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: Dataset | None = None
        self._use_streaming = False

    def load_from_hub(self, source: str, max_samples: int | None = None,
                      fetch_mode: str = "auto") -> Self:
        """Use pre-cached mel shards from a HF repo id or a local/mounted path.

        fetch_mode selects the IShardRepository: auto | local | stream | prefetch.
        auto = local if `source` is an existing directory, otherwise prefetch.
        """
        shard_repo = make_shard_repository(source, fetch_mode)
        shard_files = shard_repo.list_shards()
        print(f"  Found {len(shard_files)} shards in {source} "
              f"({type(shard_repo).__name__})")

        total = shard_repo.total_samples()
        print(f"  Total samples: {total}")

        if max_samples and max_samples < total:
            print(f"  Limiting to {max_samples} samples")
            total = max_samples

        self._shard_repo = shard_repo
        self._streaming_shards = shard_files
        self._streaming_total = total
        self._streaming_max_samples = max_samples
        self._use_streaming = True
        return self

    def load_recipe(self, recipe_name: str,
                    max_samples_per_source: int | None = None,
                    upload_repo: str | None = None,
                    samples_per_shard: int = 50000,
                    workers: int = 0) -> Self:
        """Stream multiple datasets per recipe and cache to disk.

        When upload_repo is set, shards are uploaded incrementally to HF Hub
        as they fill up — progress survives container restarts.
        """
        sources = self._repo.get_recipe(recipe_name)
        print(f"  Loading recipe '{recipe_name}': {len(sources)} sources")
        if upload_repo:
            print(f"  Incremental upload to: {upload_repo}")
        for src in sources:
            cfg_str = f" ({src['config']})" if src.get('config') else ""
            print(f"    - {src['name']} / {src['split']}{cfg_str}")

        for src in sources:
            src_cap = src.get("max_samples") or max_samples_per_source
            stream = self._repo.stream(
                src["name"], src["split"],
                config_name=src.get("config"),
                max_samples=src_cap,
                workers=workers,
            )
            label = f"{src['name']}/{src['split']}"
            count = self._cache.save_stream(
                stream, source_label=label,
                repo_id=upload_repo, samples_per_shard=samples_per_shard,
            )
            print(f"    {label}: {count} samples")

            # Free the HF datasets cache between sources to avoid filling
            # ephemeral disk (load_dataset never cleans up after itself).
            import shutil
            hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
            if os.path.isdir(hf_cache):
                shutil.rmtree(hf_cache, ignore_errors=True)
                print(f"    Cleared HF dataset cache", flush=True)

        self._use_cache = True
        return self

    def load_dataset(self, dataset_name: str, split: str | None = None,
                     config_name: str | None = None,
                     max_samples: int | None = None) -> Self:
        """Load a single dataset -- streaming or full download."""
        self._audio_col, self._text_col = self._repo.resolve_columns(dataset_name)
        print(f"  Loading dataset: {dataset_name}")
        print(f"  Columns: audio='{self._audio_col}', text='{self._text_col}'")

        if max_samples is not None:
            stream = self._repo.stream(
                dataset_name, split or "train",
                config_name=config_name, max_samples=max_samples,
            )
            self._materialized = list(stream)
            print(f"  Materialized {len(self._materialized)} samples into memory.")
            return self

        # Full download
        if split:
            self._hf_train = self._repo.fetch_full(dataset_name, split, config_name)
            print(f"  Loaded split '{split}': {len(self._hf_train)} samples")
        else:
            try:
                self._hf_train = self._repo.fetch_full(dataset_name, "train", config_name)
            except Exception:
                pass
            try:
                self._hf_val = self._repo.fetch_full(dataset_name, "validation", config_name)
            except Exception:
                pass
            try:
                self._hf_test = self._repo.fetch_full(dataset_name, "test", config_name)
            except Exception:
                pass
            sizes = []
            if self._hf_train: sizes.append(f"train={len(self._hf_train)}")
            if self._hf_val: sizes.append(f"val={len(self._hf_val)}")
            if self._hf_test: sizes.append(f"test={len(self._hf_test)}")
            print(f"  Loaded splits: {', '.join(sizes)}")
        return self

    def clean(self) -> Self:
        """Filter out empty/invalid samples from full-download datasets."""
        if self._hf_train is not None and hasattr(self._hf_train, "filter"):
            original_len = len(self._hf_train)
            a, t = self._audio_col, self._text_col
            self._hf_train = self._hf_train.filter(
                lambda x: len(x[t].strip()) > 0 and len(x[a]["array"]) > 1600
            )
            print(f"  Cleaned: {original_len} -> {len(self._hf_train)} samples")
        return self

    def preprocess(self) -> Self:
        return self

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
              max_samples: int | None = None) -> Self:
        """Split loaded data into train/val/test."""
        if self._use_streaming:
            shards = self._streaming_shards
            max_s = getattr(self, '_streaming_max_samples', None)

            # Multi-source: draw proportionally from each source for val/test
            from Shared.repositories.shard_repository import MultiShardRepository
            if isinstance(self._shard_repo, MultiShardRepository):
                by_source = self._shard_repo.shards_by_source()
                labels = self._shard_repo.labels
                train_shards, val_shards, test_shards = [], [], []
                for i, src_shards in enumerate(by_source):
                    n = len(src_shards)
                    n_tr = max(1, int(n * train_ratio))
                    n_va = max(1, int(n * val_ratio))
                    train_shards.extend(src_shards[:n_tr])
                    val_shards.extend(src_shards[n_tr:n_tr + n_va])
                    rest = src_shards[n_tr + n_va:]
                    test_shards.extend(rest or src_shards[n_tr:n_tr + n_va])
                    n_te = len(rest) or n_va
                    print(f"  Source '{labels[i]}': {n} shards → "
                          f"train={n_tr}, val={n_va}, test={n_te}")
            else:
                n = len(shards)
                n_train = max(1, int(n * train_ratio))
                n_val = max(1, int(n * val_ratio))
                train_shards = shards[:n_train]
                val_shards = shards[n_train:n_train + n_val]
                test_shards = shards[n_train + n_val:] or val_shards

            total = self._streaming_total
            sps = total // len(shards) if shards else 1

            # Distribute max_samples proportionally across splits
            train_max = int(max_s * train_ratio) if max_s else None
            val_max = int(max_s * val_ratio) if max_s else None
            test_max = (max_s - (train_max or 0) - (val_max or 0)) if max_s else None

            self._train_ds = ShardStreamingDataset(
                self._shard_repo, train_shards,
                total_samples=len(train_shards) * sps, shuffle=True,
                max_samples=train_max)
            self._val_ds = ShardStreamingDataset(
                self._shard_repo, val_shards,
                total_samples=len(val_shards) * sps, shuffle=False,
                max_samples=val_max)
            self._test_ds = ShardStreamingDataset(
                self._shard_repo, test_shards,
                total_samples=len(test_shards) * sps, shuffle=False,
                max_samples=test_max)

        elif self._use_cache:
            all_indices = self._cache.indices()
            if max_samples is not None:
                all_indices = all_indices[:max_samples]
            n = len(all_indices)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            self._train_ds = DiskCachedDataset(self._cache, all_indices[:n_train])
            self._val_ds = DiskCachedDataset(self._cache, all_indices[n_train:n_train + n_val])
            rest = all_indices[n_train + n_val:]
            self._test_ds = DiskCachedDataset(self._cache, rest if rest else all_indices[n_train:n_train + n_val])

        elif self._materialized is not None:
            samples = self._materialized
            n = len(samples)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            self._train_ds = MaterializedAudioDataset(samples[:n_train])
            self._val_ds = MaterializedAudioDataset(samples[n_train:n_train + n_val])
            rest = samples[n_train + n_val:]
            self._test_ds = MaterializedAudioDataset(rest if rest else samples[n_train:n_train + n_val])

        elif self._hf_val is not None:
            self._train_ds = HFAudioDataset(self._hf_train, self._repo, self._audio_col, self._text_col)
            self._val_ds = HFAudioDataset(self._hf_val, self._repo, self._audio_col, self._text_col)
            self._test_ds = HFAudioDataset(self._hf_test, self._repo, self._audio_col, self._text_col) if self._hf_test else self._val_ds

        elif self._hf_train is not None:
            n = len(self._hf_train)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            split_ds = self._hf_train.train_test_split(test_size=(n - n_train) / n, seed=42)
            train_hf = split_ds["train"]
            remaining = split_ds["test"]
            if len(remaining) > 1:
                val_test = remaining.train_test_split(
                    test_size=max(1, n - n_train - n_val) / len(remaining), seed=42)
                val_hf, test_hf = val_test["train"], val_test["test"]
            else:
                val_hf = test_hf = remaining
            self._train_ds = HFAudioDataset(train_hf, self._repo, self._audio_col, self._text_col)
            self._val_ds = HFAudioDataset(val_hf, self._repo, self._audio_col, self._text_col)
            self._test_ds = HFAudioDataset(test_hf, self._repo, self._audio_col, self._text_col)

        if self._train_ds:
            print(f"  Splits ready: train={len(self._train_ds)}, "
                  f"val={len(self._val_ds) if self._val_ds else 0}, "
                  f"test={len(self._test_ds) if self._test_ds else 0}")
        return self

    def _loader(self, ds: Dataset, batch_size: int, shuffle: bool,
                num_workers: int) -> DataLoader:
        is_iterable = isinstance(ds, IterableDataset)
        # IterableDataset handles its own iteration; multiple workers would
        # each download shards in parallel, blowing past ephemeral disk limits.
        workers = 0 if is_iterable else num_workers
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle and not is_iterable,
                          collate_fn=collate_fn, num_workers=workers)

    def get_train_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        assert self._train_ds is not None, "Call load_*() and split() first"
        return self._loader(self._train_ds, batch_size, shuffle=True,
                            num_workers=num_workers)

    def get_val_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        assert self._val_ds is not None, "Call load_*() and split() first"
        return self._loader(self._val_ds, batch_size, shuffle=False,
                            num_workers=num_workers)

    def get_test_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        assert self._test_ds is not None, "Call load_*() and split() first"
        return self._loader(self._test_ds, batch_size, shuffle=False,
                            num_workers=num_workers)
