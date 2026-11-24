"""
Checkpoint Service — HF Hub persistence for training checkpoints.

Handles uploading best-model weights + metadata to a HF Dataset repo
and resuming training from the last uploaded checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil

import torch


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


class CheckpointService:
    """Manages checkpoint upload/download to a HF Dataset repo."""

    def __init__(self, repo_id: str | None, save_dir: str,
                 run_name: str | None = None):
        self._repo_id = repo_id
        self._save_dir = save_dir
        self._run_name = run_name
        self._checkpoint_name = (
            f"{run_name}-best_model.pt" if run_name else "best_model.pt"
        )
        # Optimizer/scheduler/scaler state, kept separate so best_model.pt
        # stays a plain state_dict for export and eval.
        self._train_state_name = (
            f"{run_name}-train_state.pt" if run_name else "train_state.pt"
        )

    @property
    def checkpoint_name(self) -> str:
        return self._checkpoint_name

    def try_resume(self, model: torch.nn.Module, trainer=None) -> tuple[int, float]:
        """Download and load a checkpoint from the HF Dataset repo.

        If `trainer` is given, also restores optimizer/scheduler/scaler state,
        or fast-forwards the LR schedule when no saved state exists.

        Returns (start_epoch, best_val_loss). If no checkpoint found, returns (1, inf).
        """
        if not self._repo_id:
            return 1, float("inf")

        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            token = _hf_token()
            files = list_repo_files(self._repo_id, repo_type="dataset", token=token)
            ckpt_path = f"checkpoints/{self._checkpoint_name}"
            meta_path = f"checkpoints/{self._checkpoint_name}.meta.json"
            state_path = f"checkpoints/{self._train_state_name}"

            if ckpt_path not in files:
                print(f"  No checkpoint found in {self._repo_id} ({ckpt_path})", flush=True)
                return 1, float("inf")

            local_ckpt = hf_hub_download(
                self._repo_id, ckpt_path, repo_type="dataset", token=token)
            state_dict = torch.load(local_ckpt, weights_only=True, map_location="cpu")
            # Try loading as-is, then remap key names in both directions
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                remapped = {
                    k.replace(".plucker.idx_i", ".encoder.idx_i")
                     .replace(".plucker.idx_j", ".encoder.idx_j"): v
                    for k, v in state_dict.items()
                }
                try:
                    model.load_state_dict(remapped)
                except RuntimeError:
                    remapped = {
                        k.replace(".encoder.idx_i", ".plucker.idx_i")
                         .replace(".encoder.idx_j", ".plucker.idx_j"): v
                        for k, v in state_dict.items()
                    }
                    model.load_state_dict(remapped)
            del state_dict

            # Save locally so upload() finds it on the first epoch.
            os.makedirs(self._save_dir, exist_ok=True)
            shutil.copy2(local_ckpt, os.path.join(self._save_dir, "last_model.pt"))

            start_epoch = 1
            best_val_loss = float("inf")
            if meta_path in files:
                local_meta = hf_hub_download(
                    self._repo_id, meta_path, repo_type="dataset", token=token)
                with open(local_meta) as f:
                    meta = json.load(f)
                start_epoch = meta.get("epoch", 0) + 1
                best_val_loss = meta.get("val_loss", float("inf"))

            print(f"  Resumed from checkpoint: epoch {start_epoch - 1}, "
                  f"val_loss={best_val_loss:.4f}", flush=True)
            if torch.cuda.is_available():
                model.cuda()

            if trainer is not None:
                if state_path in files:
                    local_state = hf_hub_download(
                        self._repo_id, state_path, repo_type="dataset", token=token)
                    trainer.load_state_dict(
                        torch.load(local_state, weights_only=True, map_location="cpu"))
                    print("  Restored optimizer/scheduler/scaler state", flush=True)
                else:
                    trainer.fast_forward_schedule(start_epoch - 1)
                    print(f"  No saved optimizer state; LR schedule fast-forwarded "
                          f"{start_epoch - 1} epochs", flush=True)
                if trainer.scheduler is not None:
                    lr = trainer.optimizer.param_groups[0]["lr"]
                    print(f"  LR at resume: {lr:.2e}", flush=True)

            return start_epoch, best_val_loss

        except Exception as e:
            print(f"  Could not resume checkpoint: {e}", flush=True)
            return 1, float("inf")

    def _save_latest(self, model: torch.nn.Module) -> str:
        """Save the live model weights to last_model.pt and return the path."""
        os.makedirs(self._save_dir, exist_ok=True)
        path = os.path.join(self._save_dir, "last_model.pt")
        # Save to a temp file first, then rename, so a crash mid-write
        # doesn't corrupt the checkpoint.
        tmp = path + ".tmp"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
        return path

    def _archive_epoch(self, epoch: int, val_loss: float,
                       model: torch.nn.Module | None = None,
                       config=None) -> str | None:
        """Save a per-epoch copy of weights + ONNX into an archive directory.

        Layout: {save_dir}/archive/{run}-epoch_{epoch:03d}/
            model.pt, config.json, meta.json, model.onnx
        Returns the archive directory path, or None on failure.
        """
        run_prefix = f"{self._run_name}-" if self._run_name else ""
        archive_dir = os.path.join(
            self._save_dir, "archive", f"{run_prefix}epoch_{epoch:03d}")
        os.makedirs(archive_dir, exist_ok=True)

        local_pt = os.path.join(self._save_dir, "last_model.pt")
        if not os.path.isfile(local_pt):
            return None

        shutil.copy2(local_pt, os.path.join(archive_dir, "model.pt"))

        meta = {"epoch": epoch, "val_loss": val_loss,
                "run_name": self._run_name}
        with open(os.path.join(archive_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if config is not None:
            from dataclasses import asdict
            with open(os.path.join(archive_dir, "config.json"), "w") as f:
                json.dump(asdict(config), f, indent=2)

        if model is not None and config is not None:
            try:
                from Shared.services.export_service import export_ctc
                export_ctc(model, config, archive_dir)
                print(f"    -> ONNX archived: {archive_dir}/model.onnx", flush=True)
            except Exception as e:
                print(f"    -> ONNX archive failed: {e}", flush=True)

        return archive_dir

    def upload(self, epoch: int, val_loss: float, trainer=None,
               model: torch.nn.Module | None = None,
               config=None) -> None:
        """Save and upload the current model weights + metadata to HF.

        Saves a fresh last_model.pt from the live model every epoch so the
        upload always reflects the actual training state.
        """
        if model is not None:
            self._save_latest(model)

        archive_dir = self._archive_epoch(epoch, val_loss, model=model,
                                          config=config)
        if archive_dir:
            print(f"    -> Epoch {epoch} archived: {archive_dir}", flush=True)

        if not self._repo_id:
            return

        try:
            from huggingface_hub import HfApi
            token = _hf_token()
            api = HfApi(token=token)

            local_pt = os.path.join(self._save_dir, "last_model.pt")
            if not os.path.isfile(local_pt):
                return

            # Upload "latest" checkpoint (overwritten each epoch for resume)
            api.upload_file(
                path_or_fileobj=local_pt,
                path_in_repo=f"checkpoints/{self._checkpoint_name}",
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"Checkpoint epoch {epoch}: val_loss={val_loss:.4f}",
            )

            # Upload per-epoch copy (never overwritten)
            run_prefix = f"{self._run_name}-" if self._run_name else ""
            api.upload_file(
                path_or_fileobj=local_pt,
                path_in_repo=f"checkpoints/history/{run_prefix}epoch_{epoch:03d}.pt",
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"Archive epoch {epoch}: val_loss={val_loss:.4f}",
            )

            if trainer is not None:
                local_state = os.path.join(self._save_dir, self._train_state_name)
                torch.save(trainer.state_dict(), local_state)
                api.upload_file(
                    path_or_fileobj=local_state,
                    path_in_repo=f"checkpoints/{self._train_state_name}",
                    repo_id=self._repo_id,
                    repo_type="dataset",
                    commit_message=f"Train state epoch {epoch}",
                )

            # Upload archived ONNX if it was generated
            if archive_dir:
                onnx_path = os.path.join(archive_dir, "model.onnx")
                if os.path.isfile(onnx_path):
                    api.upload_file(
                        path_or_fileobj=onnx_path,
                        path_in_repo=f"checkpoints/history/{run_prefix}epoch_{epoch:03d}.onnx",
                        repo_id=self._repo_id,
                        repo_type="dataset",
                        commit_message=f"ONNX epoch {epoch}: val_loss={val_loss:.4f}",
                    )

            # Meta goes last: resume trusts it, so it should only point at an
            # epoch whose weights (and state) are already uploaded.
            meta = {"epoch": epoch, "val_loss": val_loss,
                    "checkpoint_name": self._checkpoint_name}
            meta_path = os.path.join(self._save_dir, f"{self._checkpoint_name}.meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            api.upload_file(
                path_or_fileobj=meta_path,
                path_in_repo=f"checkpoints/{self._checkpoint_name}.meta.json",
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"Meta epoch {epoch}: val_loss={val_loss:.4f}",
            )

            print(f"    -> Checkpoint uploaded to {self._repo_id} (epoch {epoch})",
                  flush=True)
        except Exception as e:
            print(f"    -> Checkpoint upload failed: {e}", flush=True)
