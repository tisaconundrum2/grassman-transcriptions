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
            # Remap legacy key names (encoder.idx_i → plucker.idx_i)
            state_dict = {
                k.replace(".encoder.idx_i", ".plucker.idx_i")
                 .replace(".encoder.idx_j", ".plucker.idx_j"): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
            del state_dict

            # Save locally so training can continue saving to same path
            os.makedirs(self._save_dir, exist_ok=True)
            shutil.copy2(local_ckpt, os.path.join(self._save_dir, "best_model.pt"))

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

    def upload(self, epoch: int, val_loss: float, trainer=None) -> None:
        """Upload best_model.pt + metadata (+ trainer state) to HF Dataset repo."""
        if not self._repo_id:
            return

        try:
            from huggingface_hub import HfApi
            token = _hf_token()
            api = HfApi(token=token)

            local_pt = os.path.join(self._save_dir, "best_model.pt")
            if not os.path.isfile(local_pt):
                return

            api.upload_file(
                path_or_fileobj=local_pt,
                path_in_repo=f"checkpoints/{self._checkpoint_name}",
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"Checkpoint epoch {epoch}: val_loss={val_loss:.4f}",
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
