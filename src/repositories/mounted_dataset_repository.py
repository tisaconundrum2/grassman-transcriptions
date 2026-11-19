"""
Mounted Dataset Repository — read raw audio from HF Jobs volume mounts.

When datasets are mounted with `-v hf://datasets/<repo>:/mnt/<name>:ro`,
the files appear as a local directory. `datasets.load_dataset()` can read
them directly — no streaming download, no HF Hub API calls. The lazy
mount fetch handles the network transparently.

Usage:
    repo = MountedDatasetRepository(config, tokenizer, mount_dir="/mnt")
    # For recipe source {"name": "openslr/librispeech_asr", "split": "train.clean.100"}
    # it loads from /mnt/librispeech_asr (last path component of the HF repo).
"""

from __future__ import annotations

import os
from typing import Iterator

import torch

from src.models.config import GrassmannConfig
from src.repositories.dataset_repository import DatasetRepository, normalize_transcript
from src.services.tokenizer_service import CharTokenizer


class MountedDatasetRepository(DatasetRepository):
    """Like DatasetRepository, but loads datasets from local mount paths.

    Maps HF dataset names to mount paths automatically:
        openslr/librispeech_asr -> {mount_dir}/librispeech_asr
        facebook/voxpopuli      -> {mount_dir}/voxpopuli
        MLCommons/peoples_speech -> {mount_dir}/peoples_speech
    """

    def __init__(self, config: GrassmannConfig, tokenizer: CharTokenizer,
                 mount_dir: str, recipes_path: str = "recipes.yaml"):
        super().__init__(config, tokenizer, recipes_path)
        self._mount_dir = mount_dir

    def _resolve_mount_path(self, dataset_name: str) -> str:
        """Map 'owner/repo' to '{mount_dir}/repo'."""
        repo_name = dataset_name.split("/")[-1]
        path = os.path.join(self._mount_dir, repo_name)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Mount not found: {path} (expected from dataset '{dataset_name}'). "
                f"Add -v hf://datasets/{dataset_name}:{path}:ro to the job command."
            )
        return path

    def stream(self, dataset_name: str, split: str,
               config_name: str | None = None,
               max_samples: int | None = None,
               audio_col: str | None = None,
               text_col: str | None = None,
               workers: int | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Load from mounted path and yield (mel, tokens) pairs."""
        from datasets import load_dataset

        mount_path = self._resolve_mount_path(dataset_name)
        a_col, t_col = self.resolve_columns(dataset_name, audio_col, text_col)
        needs_normalize = "librispeech" not in dataset_name.lower()

        print(f"      Loading from mount: {mount_path} (config={config_name}, split={split})",
              flush=True)
        ds = load_dataset(mount_path, config_name, split=split)
        print(f"      Loaded {len(ds)} samples from mount", flush=True)

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
