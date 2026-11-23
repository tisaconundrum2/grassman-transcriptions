"""
ModelController: fluent API for model lifecycle.

Provides the chainable interface:
    _model.build().train(...).test(...).save(...)
"""

from __future__ import annotations

import os
from typing import Any, Self, Type

import torch
import torch.nn as nn

from Shared.interfaces import IModel
from Shared.models.config import GrassmannConfig
from Shared.models.grassmann_model import GrassmannTranscriptionModel
from Shared.services.tokenizer_service import CharTokenizer
from Shared.services.training_service import TrainingService
from Shared.views.console_view import ConsoleView


class ModelController(IModel):
    """Orchestrates model building, training, evaluation, and inference."""

    def __init__(self, config: GrassmannConfig, tokenizer: CharTokenizer,
                 layer_cls: Type[nn.Module] | None = None,
                 block_cls: Type[nn.Module] | None = None):
        self.config = config
        self.tokenizer = tokenizer
        self._layer_cls = layer_cls
        self._block_cls = block_cls
        self.view = ConsoleView()
        self.nn_model: GrassmannTranscriptionModel | None = None
        self.trainer: TrainingService | None = None

    def build(self) -> Self:
        self.config.vocab_size = self.tokenizer.vocab_size

        self.nn_model = GrassmannTranscriptionModel(
            self.config,
            layer_cls=self._layer_cls,
            block_cls=self._block_cls,
        )
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

    def decode_samples(self, val_ds: Any, n: int = 3) -> None:
        """Decode and print sample transcriptions from a validation dataset."""
        assert self.trainer is not None
        print("\n  Sample transcriptions:", flush=True)
        # Support both map-style (indexable) and iterable datasets
        from torch.utils.data import IterableDataset
        if isinstance(val_ds, IterableDataset):
            for i, (mel, tokens) in enumerate(val_ds):
                if i >= n:
                    break
                pred_text = self.predict(mel)
                gt_text = self.tokenizer.decode(tokens)
                print(f"    [{i}] GT:   '{gt_text}'", flush=True)
                print(f"    [{i}] PRED: '{pred_text}'")
        else:
            for i in range(min(n, len(val_ds))):
                mel, tokens = val_ds[i]
                pred_text = self.predict(mel)
                gt_text = self.tokenizer.decode(tokens)
                print(f"    [{i}] GT:   '{gt_text}'", flush=True)
                print(f"    [{i}] PRED: '{pred_text}'")
        print()

    def post_training(self, save_dir: str, run_name: str | None = None) -> None:
        """Bundle model, export ONNX, rename with run_name, and upload."""
        import shutil
        from Shared.services.export_service import export_seq2seq, export_ctc

        def _prefixed(name: str) -> str:
            if not run_name:
                return name
            base, ext = os.path.splitext(name)
            return f"{run_name}-{base}{ext}"

        # Save bundle
        try:
            from Shared.services.model_hub_service import save_model
            assert self.nn_model is not None
            save_model(self.nn_model, self.config, os.path.join(save_dir, "bundle"))
            print("  Model bundle saved.", flush=True)
        except Exception as e:
            print(f"  Warning: bundle save failed: {e}", flush=True)

        # Export ONNX
        try:
            onnx_dir = os.path.join(save_dir, "onnx")
            if self.config.use_seq2seq:
                export_seq2seq(self.nn_model, self.config, onnx_dir)
            else:
                export_ctc(self.nn_model, self.config, onnx_dir)
            print(f"  ONNX exported to {onnx_dir}/", flush=True)
        except Exception as e:
            print(f"  Warning: ONNX export failed: {e}", flush=True)

        # Rename artifacts with run_name prefix
        if run_name:
            for name in ["best_model.pt", "final_model.pt"]:
                src = os.path.join(save_dir, name)
                dst = os.path.join(save_dir, _prefixed(name))
                if os.path.isfile(src):
                    shutil.move(src, dst)
            onnx_dir = os.path.join(save_dir, "onnx")
            if os.path.isdir(onnx_dir):
                for fname in list(os.listdir(onnx_dir)):
                    src = os.path.join(onnx_dir, fname)
                    dst = os.path.join(onnx_dir, _prefixed(fname))
                    if os.path.isfile(src):
                        shutil.move(src, dst)
            bundle_dir = os.path.join(save_dir, "bundle")
            if os.path.isdir(bundle_dir):
                for fname in list(os.listdir(bundle_dir)):
                    src = os.path.join(bundle_dir, fname)
                    dst = os.path.join(bundle_dir, _prefixed(fname))
                    if os.path.isfile(src):
                        shutil.move(src, dst)

        # Upload
        try:
            from Shared.services.upload_service import upload_file, upload_directory
            bundle_dir = os.path.join(save_dir, "bundle")
            onnx_dir = os.path.join(save_dir, "onnx")
            if os.path.isdir(bundle_dir):
                for fname in sorted(os.listdir(bundle_dir)):
                    fpath = os.path.join(bundle_dir, fname)
                    if os.path.isfile(fpath):
                        upload_file(fpath)
            if os.path.isdir(onnx_dir):
                urls = upload_directory(onnx_dir)
                print(f"  Uploaded {len(urls)} ONNX files", flush=True)
            best_name = _prefixed("best_model.pt")
            best_path = os.path.join(save_dir, best_name)
            if os.path.isfile(best_path):
                upload_file(best_path)
        except Exception as e:
            print(f"  Warning: upload failed: {e}", flush=True)
