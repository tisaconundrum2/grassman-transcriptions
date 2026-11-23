"""
Dataset Repository — external data access for HuggingFace audio datasets.

Handles fetching from HF Hub, schema resolution, streaming, and
mel spectrogram extraction. Supports concurrent mel transforms via
ProcessPoolExecutor for faster caching on multi-core machines.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

import torch
import torchaudio.transforms as T
import yaml

from Shared.models.config import GrassmannConfig
from Shared.services.tokenizer_service import CharTokenizer


def _process_raw_sample(args: tuple) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Process a single (waveform_array, sr, text, needs_normalize) in a worker.

    Top-level function so it's picklable by ProcessPoolExecutor.
    Returns (mel, tokens) or None if invalid.
    """
    waveform_array, sr, text, needs_normalize, config_dict = args

    # Reconstruct transforms in worker (not picklable across processes)
    target_sr = config_dict["sample_rate"]
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr, n_fft=config_dict["n_fft"],
        hop_length=config_dict["hop_length"], n_mels=config_dict["n_mels"],
    )

    try:
        waveform = torch.tensor(waveform_array, dtype=torch.float32)
        if sr != target_sr:
            waveform = T.Resample(sr, target_sr)(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-9)).squeeze(0)
        mean = mel.mean()
        std = mel.std().clamp(min=1e-9)
        mel = (mel - mean) / std

        if needs_normalize:
            text = normalize_transcript(text)

        # Inline tokenization (CharTokenizer is simple enough)
        vocab = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '")}
        tokens = torch.tensor([vocab.get(c, 0) for c in text.lower()], dtype=torch.long)
        if len(tokens) == 0:
            return None

        return mel, tokens
    except Exception:
        return None


def _load_recipes_config(config_path: str = "recipes.yaml") -> dict:
    """Load dataset schemas and recipes from YAML config."""
    path = Path(config_path)
    if not path.exists():
        return {"schemas": {}, "recipes": {}}
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_transcript(text: str) -> str:
    """Normalize transcript to lowercase ASCII with only letters, spaces, apostrophes."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class DatasetRepository:
    """Fetches audio+text samples from HuggingFace datasets.

    Resolves column schemas automatically from recipes.yaml or known defaults.
    Streams samples as (mel, tokens) pairs — the caller decides where they go
    (memory, disk cache, etc.).

    When input_dir is set, datasets are loaded from local/mounted paths
    (e.g. /mnt/librispeech_asr) instead of streaming from the Hub.
    """

    def __init__(self, config: GrassmannConfig, tokenizer: CharTokenizer,
                 recipes_path: str = "recipes.yaml",
                 input_dir: str | None = None):
        self.config = config
        self.tokenizer = tokenizer
        self._input_dir = input_dir
        self._recipes_config = _load_recipes_config(recipes_path)
        self._schemas: dict[str, dict[str, str]] = self._recipes_config.get("schemas", {})
        self._recipes: dict[str, list[dict]] = self._recipes_config.get("recipes", {})
        self._mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels,
        )

    def resolve_columns(self, dataset_name: str,
                        audio_col: str | None = None,
                        text_col: str | None = None) -> tuple[str, str]:
        """Resolve audio and text column names for a dataset."""
        if audio_col and text_col:
            return audio_col, text_col
        for known_name, schema in self._schemas.items():
            if known_name in dataset_name:
                return audio_col or schema["audio"], text_col or schema["text"]
        return audio_col or "audio", text_col or "text"

    def get_recipe(self, name: str) -> list[dict]:
        """Get a named recipe from config."""
        if name not in self._recipes:
            available = list(self._recipes.keys())
            raise ValueError(f"Unknown recipe '{name}'. Available: {available}")
        return self._recipes[name]

    @property
    def available_recipes(self) -> list[str]:
        return list(self._recipes.keys())

    def process_sample(self, sample: dict, audio_col: str,
                       text_col: str) -> tuple[torch.Tensor, str]:
        """Extract mel spectrogram and text from a single HF sample."""
        audio_data = sample[audio_col]
        waveform = torch.tensor(audio_data["array"], dtype=torch.float32)
        sr = audio_data["sampling_rate"]

        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self._mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-9))
        mel = mel.squeeze(0)

        mean = mel.mean()
        std = mel.std().clamp(min=1e-9)
        mel = (mel - mean) / std

        return mel, sample[text_col]

    def stream(self, dataset_name: str, split: str,
               config_name: str | None = None,
               max_samples: int | None = None,
               audio_col: str | None = None,
               text_col: str | None = None,
               workers: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Stream (mel, tokens) pairs from a HF dataset.

        Uses ProcessPoolExecutor to compute mel spectrograms concurrently.
        Workers defaults to CPU count. Set workers=0 for sequential mode.
        """
        from datasets import load_dataset

        a_col, t_col = self.resolve_columns(dataset_name, audio_col, text_col)
        needs_normalize = "librispeech" not in dataset_name.lower()
        n_workers = workers if workers is not None else min(os.cpu_count() or 1, 8)

        ds = self._load(dataset_name, config_name, split)
        count = 0

        # Config dict for worker processes (can't pickle MelSpectrogram)
        config_dict = {
            "sample_rate": self.config.sample_rate,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "n_mels": self.config.n_mels,
        }

        # Sequential fallback
        if n_workers < 2:
            yield from self._stream_sequential(
                ds, a_col, t_col, needs_normalize, max_samples)
            return

        # Concurrent: prefetch batches of raw samples, fan out mel transforms
        batch_size = n_workers * 4  # keep workers fed
        raw_batch: list[tuple] = []

        for sample in ds:
            if max_samples and count >= max_samples:
                break

            try:
                raw_text = sample[t_col].strip()
            except (KeyError, AttributeError):
                continue

            text = normalize_transcript(raw_text) if needs_normalize else raw_text
            if len(text) == 0:
                continue

            try:
                audio_array = sample[a_col]["array"]
            except (KeyError, TypeError):
                continue

            if len(audio_array) < 1600 or len(audio_array) > 240000:
                continue

            sr = sample[a_col]["sampling_rate"]
            raw_batch.append((audio_array, sr, text, needs_normalize, config_dict))

            if len(raw_batch) >= batch_size:
                for result in self._process_batch(raw_batch, n_workers):
                    if max_samples and count >= max_samples:
                        break
                    yield result
                    count += 1
                raw_batch = []

        # Drain remaining batch
        if raw_batch and (not max_samples or count < max_samples):
            for result in self._process_batch(raw_batch, n_workers):
                if max_samples and count >= max_samples:
                    break
                yield result
                count += 1

        del ds

    def _process_batch(self, raw_batch: list[tuple],
                       n_workers: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Process a batch of raw samples concurrently."""
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for result in pool.map(_process_raw_sample, raw_batch):
                if result is not None:
                    results.append(result)
        return results

    def _stream_sequential(self, ds, a_col: str, t_col: str,
                           needs_normalize: bool,
                           max_samples: int | None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Sequential fallback for stream()."""
        count = 0
        for sample in ds:
            if max_samples and count >= max_samples:
                break

            try:
                raw_text = sample[t_col].strip()
            except (KeyError, AttributeError):
                continue

            text = normalize_transcript(raw_text) if needs_normalize else raw_text
            if len(text) == 0:
                continue

            try:
                audio_array = sample[a_col]["array"]
            except (KeyError, TypeError):
                continue

            if len(audio_array) < 1600 or len(audio_array) > 240000:
                continue

            try:
                mel, _ = self.process_sample(sample, a_col, t_col)
            except Exception:
                continue

            if needs_normalize:
                text = normalize_transcript(raw_text)
            tokens = self.tokenizer.encode(text)
            if len(tokens) == 0:
                continue

            yield mel, tokens
            count += 1

        del ds

    def _load(self, dataset_name: str, config_name: str | None, split: str):
        """Load a dataset from local mount or Hub — always streaming.

        Streaming avoids building Arrow caches on disk, which is critical for
        mounted volumes where load_dataset would otherwise cache every split's
        parquet files and blow past the ephemeral disk limit.
        """
        from datasets import load_dataset
        if self._input_dir:
            repo_name = dataset_name.split("/")[-1]
            local_path = os.path.join(self._input_dir, repo_name)
            if not os.path.isdir(local_path):
                raise FileNotFoundError(
                    f"Mount not found: {local_path} "
                    f"(expected from dataset '{dataset_name}'). "
                    f"Add -v hf://datasets/{dataset_name}:{local_path}:ro")
            print(f"      Streaming from mount: {local_path} (config={config_name}, "
                  f"split={split})", flush=True)
            return load_dataset(local_path, config_name, split=split, streaming=True)
        return load_dataset(dataset_name, config_name, split=split, streaming=True)

    def fetch_full(self, dataset_name: str, split: str,
                   config_name: str | None = None):
        """Download a full (non-streaming) HF dataset split. Returns the HF dataset object."""
        from datasets import load_dataset
        return load_dataset(dataset_name, config_name, split=split)
