from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from src.interfaces import IDataset
from src.models.config import GrassmannConfig
from src.services.audio_service import AudioService
from src.services.tokenizer_service import CharTokenizer


class AudioTranscriptionDataset(Dataset):
    """PyTorch Dataset for (mel_spectrogram, transcript) pairs."""

    def __init__(
        self,
        audio_paths: list[str],
        transcripts: list[str],
        audio_service: AudioService,
        tokenizer: CharTokenizer,
    ):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.audio_service = audio_service
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform = self.audio_service.load(self.audio_paths[idx])
        mel = self.audio_service.extract_features(waveform)
        mel = self.audio_service.normalize(mel)
        tokens = self.tokenizer.encode(self.transcripts[idx])
        return mel, tokens


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length mel spectrograms and token sequences."""
    mels, tokens = zip(*batch)

    # Pad mels along time dimension
    mel_lengths = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
    max_mel_len = mel_lengths.max().item()
    padded_mels = []
    for m in mels:
        pad_amount = max_mel_len - m.shape[-1]
        padded_mels.append(torch.nn.functional.pad(m, (0, pad_amount)))
    mel_batch = torch.stack(padded_mels)  # (B, n_mels, T)

    # Pad token sequences
    token_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    token_batch = pad_sequence(tokens, batch_first=True, padding_value=0)

    return mel_batch, token_batch, mel_lengths, token_lengths


class DatasetService(IDataset):
    """Manages dataset loading, cleaning, and DataLoader creation."""

    def __init__(self, config: GrassmannConfig, audio_service: AudioService, tokenizer: CharTokenizer):
        self.config = config
        self.audio_service = audio_service
        self.tokenizer = tokenizer
        self.audio_paths: list[str] = []
        self.transcripts: list[str] = []
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: Dataset | None = None

    def load(self, path: str) -> Self:
        """
        Load dataset from a directory with structure:
            path/
              metadata.csv   (filename|transcript per line)
              wavs/
                file1.wav
                ...
        """
        metadata_file = Path(path) / "metadata.csv"
        wavs_dir = Path(path) / "wavs"

        if metadata_file.exists():
            with open(metadata_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|", maxsplit=1)
                    if len(parts) == 2:
                        fname, transcript = parts
                        audio_path = str(wavs_dir / f"{fname}.wav")
                        if os.path.exists(audio_path):
                            self.audio_paths.append(audio_path)
                            self.transcripts.append(transcript)
        return self

    def load_from_pairs(self, pairs: list[tuple[str, str]]) -> Self:
        """Load from explicit (audio_path, transcript) pairs."""
        for audio_path, transcript in pairs:
            self.audio_paths.append(audio_path)
            self.transcripts.append(transcript)
        return self

    def clean(self) -> Self:
        """Filter out invalid entries."""
        valid_paths = []
        valid_transcripts = []
        for p, t in zip(self.audio_paths, self.transcripts):
            if os.path.exists(p) and len(t.strip()) > 0:
                valid_paths.append(p)
                valid_transcripts.append(t.strip().lower())
        self.audio_paths = valid_paths
        self.transcripts = valid_transcripts
        return self

    def preprocess(self) -> Self:
        """No-op for now; feature extraction happens on-the-fly in Dataset."""
        return self

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Self:
        full_ds = AudioTranscriptionDataset(
            self.audio_paths, self.transcripts, self.audio_service, self.tokenizer
        )
        n = len(full_ds)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        self._train_ds, self._val_ds, self._test_ds = random_split(
            full_ds, [n_train, n_val, n_test]
        )
        return self

    def get_train_loader(self, batch_size: int) -> DataLoader:
        assert self._train_ds is not None, "Call split() first"
        return DataLoader(
            self._train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )

    def get_val_loader(self, batch_size: int) -> DataLoader:
        assert self._val_ds is not None, "Call split() first"
        return DataLoader(
            self._val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

    def get_test_loader(self, batch_size: int) -> DataLoader:
        assert self._test_ds is not None, "Call split() first"
        return DataLoader(
            self._test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
