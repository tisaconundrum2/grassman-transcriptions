"""
PipelineController: end-to-end orchestration.

Wires together dataset, model, and services into a single fluent pipeline.
"""

from __future__ import annotations

from typing import Self

from src.interfaces import IDataset
from src.models.config import GrassmannConfig
from src.services.audio_service import AudioService
from src.services.dataset_service import DatasetService
from src.services.tokenizer_service import CharTokenizer
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.cache_repository import CacheRepository
from src.controllers.model_controller import ModelController


class PipelineController:
    """
    Top-level controller for the full transcription pipeline.

    Usage (local files):
        pipeline = PipelineController(config)
        pipeline.prepare_dataset("path/to/data")
                .build_model()
                .train()
                .test()
                .transcribe("path/to/audio.wav")

    Usage (HuggingFace):
        pipeline = PipelineController(config)
        pipeline.prepare_hf_dataset("openslr/librispeech_asr",
                                     split="train.clean.100")
                .build_model()
                .train()
                .test()
    """

    def __init__(self, config: GrassmannConfig | None = None,
                 cache_dir: str = "checkpoints/mel_cache"):
        self.config = config or GrassmannConfig()
        self.tokenizer = CharTokenizer()
        self.audio_service = AudioService(self.config)
        self._repo = DatasetRepository(self.config, self.tokenizer)
        self._cache = CacheRepository(cache_dir)
        self.dataset: DatasetService | None = None
        self.model: ModelController | None = None

    def prepare_hf_dataset(
        self,
        dataset_name: str,
        config_name: str | None = None,
        split: str | None = None,
        max_samples: int | None = None,
    ) -> Self:
        """Load a dataset from HuggingFace Hub."""
        self.dataset = DatasetService(self._repo, self._cache)
        self.dataset.load_dataset(
            dataset_name, split=split,
            config_name=config_name,
            max_samples=max_samples,
        ).clean().preprocess().split()
        return self

    def build_model(self) -> Self:
        self.model = ModelController(self.config, self.tokenizer)
        self.model.build()
        return self

    def train(self, epochs: int | None = None) -> Self:
        assert self.model is not None and self.dataset is not None
        train_loader = self.dataset.get_train_loader(self.config.batch_size)
        val_loader = self.dataset.get_val_loader(self.config.batch_size)
        self.model.train(train_loader, val_loader, epochs)
        return self

    def test(self) -> dict:
        assert self.model is not None and self.dataset is not None
        test_loader = self.dataset.get_test_loader(self.config.batch_size)
        return self.model.test(test_loader)

    def transcribe(self, audio_path: str) -> str:
        assert self.model is not None
        waveform = self.audio_service.load(audio_path)
        mel = self.audio_service.extract_features(waveform)
        mel = self.audio_service.normalize(mel)
        return self.model.predict(mel)

    def save_model(self, path: str) -> Self:
        assert self.model is not None
        self.model.save(path)
        return self

    def load_model(self, path: str) -> Self:
        if self.model is None:
            self.build_model()
        assert self.model is not None
        self.model.load(path)
        return self
