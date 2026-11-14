"""
ModelController: fluent API for model lifecycle.

Provides the chainable interface:
    _model.build().train(...).test(...).save(...)
"""

from __future__ import annotations

from typing import Any, Self

import torch

from src.interfaces import IModel
from src.models.config import GrassmannConfig
from src.models.grassmann_model import GrassmannTranscriptionModel
from src.models.grassmann_seq2seq import GrassmannSeq2SeqModel
from src.services.tokenizer_service import CharTokenizer
from src.services.training_service import TrainingService
from src.services.seq2seq_training_service import Seq2SeqTrainingService
from src.views.console_view import ConsoleView


class ModelController(IModel):
    """Orchestrates model building, training, evaluation, and inference."""

    def __init__(self, config: GrassmannConfig, tokenizer: CharTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.view = ConsoleView()
        self.nn_model: GrassmannTranscriptionModel | GrassmannSeq2SeqModel | None = None
        self.trainer: TrainingService | Seq2SeqTrainingService | None = None

    def build(self) -> Self:
        self.config.vocab_size = self.tokenizer.vocab_size

        if self.config.use_seq2seq:
            self.nn_model = GrassmannSeq2SeqModel(self.config)
            self.trainer = Seq2SeqTrainingService(self.nn_model, self.config)
        else:
            self.nn_model = GrassmannTranscriptionModel(self.config)
            self.trainer = TrainingService(self.nn_model, self.config)

        param_count = sum(p.numel() for p in self.nn_model.parameters())
        self.view.show_model_info(self.config, param_count)
        return self

    def train(self, train_loader: Any, val_loader: Any, epochs: int | None = None) -> Self:
        assert self.trainer is not None, "Call build() first"
        epochs = epochs or self.config.epochs

        history = self.trainer.train_loop(
            train_loader, val_loader, epochs, callback=self.view
        )
        self.view.show_training_complete(history)
        return self

    def test(self, test_loader: Any) -> dict:
        assert self.trainer is not None, "Call build() and train() first"
        test_loss = self.trainer.validate(test_loader)
        results = {"test_loss": test_loss}
        self.view.show_test_results(results)
        return results

    def fit(self, dataset: Any) -> Self:
        """Convenience: build + train + validate."""
        self.build()

        train_loader = dataset.get_train_loader(self.config.batch_size)
        val_loader = dataset.get_val_loader(self.config.batch_size)

        self.train(train_loader, val_loader)
        return self

    def predict(self, audio: torch.Tensor) -> str:
        assert self.trainer is not None, "Call build() and train()/load() first"
        pred_tokens = self.trainer.greedy_decode(audio)
        text = self.tokenizer.decode(pred_tokens)
        return text

    def save(self, path: str) -> Self:
        assert self.nn_model is not None
        torch.save(self.nn_model.state_dict(), path)
        self.view.show_message(f"Model saved to {path}")
        return self

    def load(self, path: str) -> Self:
        if self.nn_model is None:
            self.build()
        assert self.nn_model is not None
        self.nn_model.load_state_dict(torch.load(path, weights_only=True))
        self.view.show_message(f"Model loaded from {path}")
        return self
