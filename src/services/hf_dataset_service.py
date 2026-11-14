"""
HuggingFace Dataset Service.

Loads any HuggingFace audio+text dataset (LibriSpeech, Common Voice, FLEURS, etc.)
and wraps it into PyTorch DataLoaders for the Grassmann transcription pipeline.

Uses streaming mode by default when max_samples is set, so we only download
the rows we actually need instead of multi-GB parquet shards.
"""

from __future__ import annotations

from typing import Self

import torch
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.interfaces import IDataset
from src.models.config import GrassmannConfig
from src.services.tokenizer_service import CharTokenizer


KNOWN_SCHEMAS: dict[str, tuple[str, str]] = {
    "openslr/librispeech_asr": ("audio", "text"),
    "mozilla-foundation/common_voice": ("audio", "sentence"),
    "google/fleurs": ("audio", "transcription"),
    "speechcolab/gigaspeech": ("audio", "text"),
    "facebook/voxpopuli": ("audio", "normalized_text"),
    "espnet/yodas": ("audio", "text"),
}


def _resolve_columns(dataset_name: str, audio_col: str | None, text_col: str | None) -> tuple[str, str]:
    if audio_col and text_col:
        return audio_col, text_col
    for known_name, (a, t) in KNOWN_SCHEMAS.items():
        if known_name in dataset_name:
            return audio_col or a, text_col or t
    return audio_col or "audio", text_col or "text"


def _process_sample(
    sample: dict,
    audio_col: str,
    text_col: str,
    mel_transform: T.MelSpectrogram,
    target_sr: int,
) -> tuple[torch.Tensor, str]:
    """Extract mel spectrogram and text from a single HF sample."""
    audio_data = sample[audio_col]
    waveform = torch.tensor(audio_data["array"], dtype=torch.float32)
    sr = audio_data["sampling_rate"]

    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel = mel_transform(waveform)
    mel = torch.log(mel.clamp(min=1e-9))
    mel = mel.squeeze(0)

    mean = mel.mean()
    std = mel.std().clamp(min=1e-9)
    mel = (mel - mean) / std

    return mel, sample[text_col]


class MaterializedAudioDataset(Dataset):
    """Pre-materialized dataset -- samples already in memory as (mel, tokens) pairs."""

    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class DiskCachedDataset(Dataset):
    """Dataset backed by individual .pt files on disk. Low RAM usage."""

    def __init__(self, cache_dir: str, indices: list[int] | None = None):
        from pathlib import Path
        self._cache_dir = Path(cache_dir)
        # If indices provided, use subset; otherwise use all files
        if indices is not None:
            self._indices = indices
        else:
            self._indices = sorted(
                int(f.stem) for f in self._cache_dir.glob("*.pt")
            )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx = self._indices[idx]
        return torch.load(
            self._cache_dir / f"{file_idx}.pt", weights_only=True
        )


class HFAudioDataset(Dataset):
    """Wraps a (non-streaming) HuggingFace dataset split into a PyTorch Dataset."""

    def __init__(self, hf_dataset, audio_col: str, text_col: str,
                 tokenizer: CharTokenizer, config: GrassmannConfig):
        self.hf_dataset = hf_dataset
        self.audio_col = audio_col
        self.text_col = text_col
        self.tokenizer = tokenizer
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels,
        )
        self.target_sr = config.sample_rate

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.hf_dataset[idx]
        mel, text = _process_sample(
            sample, self.audio_col, self.text_col, self.mel_transform, self.target_sr
        )
        tokens = self.tokenizer.encode(text)
        return mel, tokens


def hf_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _stream_and_materialize(
    dataset_name: str,
    config_name: str | None,
    split: str,
    max_samples: int,
    audio_col: str,
    text_col: str,
    tokenizer: CharTokenizer,
    config: GrassmannConfig,
    cache_dir: str | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]] | str:
    """Stream a HF dataset and save processed samples.

    If cache_dir is set, saves each (mel, tokens) pair as a .pt file on disk
    and returns the cache_dir path (low RAM). Otherwise materializes into a list
    in memory (original behavior).
    """
    from pathlib import Path
    from datasets import load_dataset

    # Check if cache already exists
    if cache_dir:
        cache_path = Path(cache_dir)
        existing = list(cache_path.glob("*.pt"))
        if existing:
            print(f"  Found {len(existing)} cached samples in {cache_dir}, skipping stream.")
            return cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)

    print(f"  Streaming {max_samples} samples from '{split}'...")
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
    )

    import gc

    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    saved_count = 0
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break
        text = sample[text_col].strip()
        audio_array = sample[audio_col]["array"]
        # Skip empty, too-short (<0.1s), or too-long (>20s at 16kHz = 320000 samples) clips
        # Long clips cause massive GPU memory spikes during conv2d forward pass
        if len(text) == 0 or len(audio_array) < 1600 or len(audio_array) > 320000:
            del sample, audio_array
            continue
        mel, text = _process_sample(sample, audio_col, text_col, mel_transform, config.sample_rate)
        tokens = tokenizer.encode(text)
        if cache_dir:
            torch.save((mel, tokens), Path(cache_dir) / f"{saved_count}.pt")
            saved_count += 1
            del mel, tokens
        else:
            samples.append((mel, tokens))
        del sample, audio_array
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{max_samples} samples...")
        # Periodic GC during caching to prevent RAM buildup from librosa/datasets buffers
        if cache_dir and (i + 1) % 500 == 0:
            gc.collect()

    # Explicitly release the streaming dataset and its internal buffers
    del ds
    gc.collect()

    if cache_dir:
        print(f"  Cached {saved_count} samples to disk at {cache_dir}.")
        return cache_dir
    else:
        print(f"  Materialized {len(samples)} samples into memory.")
        return samples


class HFDatasetService(IDataset):
    """Loads audio datasets from HuggingFace Hub.

    Usage:
        _dataset = HFDatasetService(config, tokenizer)
        _dataset.load("openslr/librispeech_asr", split="train.clean.100",
                       max_samples=100)
               .clean().preprocess().split()
    """

    def __init__(self, config: GrassmannConfig, tokenizer: CharTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_name: str = ""
        self.audio_col: str = "audio"
        self.text_col: str = "text"
        self._hf_train = None
        self._hf_val = None
        self._hf_test = None
        self._materialized: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._cache_dir: str | None = None
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: Dataset | None = None

    def load(self, dataset_name: str, config_name: str | None = None,
             split: str | None = None, audio_col: str | None = None,
             text_col: str | None = None, streaming: bool = False,
             max_samples: int | None = None,
             cache_dir: str | None = None) -> Self:
        from datasets import load_dataset

        self.dataset_name = dataset_name
        self.audio_col, self.text_col = _resolve_columns(dataset_name, audio_col, text_col)
        print(f"  Loading dataset: {dataset_name}")
        print(f"  Columns: audio='{self.audio_col}', text='{self.text_col}'")

        # When max_samples is set, stream to avoid downloading everything
        if max_samples is not None:
            train_split = split or "train"
            result = _stream_and_materialize(
                dataset_name, config_name, train_split, max_samples,
                self.audio_col, self.text_col, self.tokenizer, self.config,
                cache_dir=cache_dir,
            )
            if isinstance(result, str):
                self._cache_dir = result
            else:
                self._materialized = result
            return self

        # Full download mode
        if split:
            ds = load_dataset(dataset_name, config_name, split=split)
            self._hf_train = ds
            print(f"  Loaded split '{split}': {len(ds)} samples")
        else:
            try:
                self._hf_train = load_dataset(dataset_name, config_name, split="train")
            except Exception:
                pass
            try:
                self._hf_val = load_dataset(dataset_name, config_name, split="validation")
            except Exception:
                pass
            try:
                self._hf_test = load_dataset(dataset_name, config_name, split="test")
            except Exception:
                pass
            sizes = []
            if self._hf_train: sizes.append(f"train={len(self._hf_train)}")
            if self._hf_val: sizes.append(f"val={len(self._hf_val)}")
            if self._hf_test: sizes.append(f"test={len(self._hf_test)}")
            print(f"  Loaded splits: {', '.join(sizes)}")
        return self

    def clean(self) -> Self:
        if self._materialized is not None:
            return self  # already filtered during streaming
        if self._hf_train is not None and hasattr(self._hf_train, "filter"):
            original_len = len(self._hf_train)
            self._hf_train = self._hf_train.filter(
                lambda x: len(x[self.text_col].strip()) > 0
                and len(x[self.audio_col]["array"]) > 1600
            )
            print(f"  Cleaned: {original_len} -> {len(self._hf_train)} samples")
        return self

    def preprocess(self) -> Self:
        return self

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Self:
        if self._cache_dir is not None:
            # Disk-cached mode: split by file indices
            from pathlib import Path
            all_indices = sorted(
                int(f.stem) for f in Path(self._cache_dir).glob("*.pt")
            )
            n = len(all_indices)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            self._train_ds = DiskCachedDataset(self._cache_dir, all_indices[:n_train])
            self._val_ds = DiskCachedDataset(self._cache_dir, all_indices[n_train:n_train + n_val])
            rest = all_indices[n_train + n_val:]
            self._test_ds = DiskCachedDataset(self._cache_dir, rest if rest else all_indices[n_train:n_train + n_val])
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
            self._train_ds = HFAudioDataset(self._hf_train, self.audio_col, self.text_col, self.tokenizer, self.config)
            self._val_ds = HFAudioDataset(self._hf_val, self.audio_col, self.text_col, self.tokenizer, self.config)
            self._test_ds = HFAudioDataset(self._hf_test, self.audio_col, self.text_col, self.tokenizer, self.config) if self._hf_test else self._val_ds
        elif self._hf_train is not None:
            n = len(self._hf_train)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            split_ds = self._hf_train.train_test_split(test_size=(n - n_train) / n, seed=42)
            train_hf = split_ds["train"]
            remaining = split_ds["test"]
            if len(remaining) > 1:
                val_test = remaining.train_test_split(test_size=max(1, n - n_train - n_val) / len(remaining), seed=42)
                val_hf, test_hf = val_test["train"], val_test["test"]
            else:
                val_hf = test_hf = remaining
            self._train_ds = HFAudioDataset(train_hf, self.audio_col, self.text_col, self.tokenizer, self.config)
            self._val_ds = HFAudioDataset(val_hf, self.audio_col, self.text_col, self.tokenizer, self.config)
            self._test_ds = HFAudioDataset(test_hf, self.audio_col, self.text_col, self.tokenizer, self.config)

        if self._train_ds:
            print(f"  Splits ready: train={len(self._train_ds)}, "
                  f"val={len(self._val_ds) if self._val_ds else 0}, "
                  f"test={len(self._test_ds) if self._test_ds else 0}")
        return self

    def get_train_loader(self, batch_size: int) -> DataLoader:
        assert self._train_ds is not None, "Call load() and split() first"
        return DataLoader(self._train_ds, batch_size=batch_size, shuffle=True, collate_fn=hf_collate_fn, num_workers=0)

    def get_val_loader(self, batch_size: int) -> DataLoader:
        assert self._val_ds is not None, "Call load() and split() first"
        return DataLoader(self._val_ds, batch_size=batch_size, shuffle=False, collate_fn=hf_collate_fn, num_workers=0)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        assert self._test_ds is not None, "Call load() and split() first"
        return DataLoader(self._test_ds, batch_size=batch_size, shuffle=False, collate_fn=hf_collate_fn, num_workers=0)
