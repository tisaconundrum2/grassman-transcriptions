"""
Cache Repository — disk and HF Hub persistence for pre-computed mel spectrograms.

Handles reading/writing individual .pt files on local disk, packing/unpacking
shards for HF Hub upload/download. Supports incremental shard uploads so
progress survives container restarts on ephemeral environments (HF Spaces).
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Iterator

import torch


def _hf_token() -> str | None:
    """Get HF token from environment."""
    return os.environ.get("HF_TOKEN")


class CacheRepository:
    """Manages a local disk cache of (mel, tokens) .pt files and
    syncs with a HuggingFace Dataset repo for persistence."""

    def __init__(self, cache_dir: str):
        self._cache_dir = Path(cache_dir)
        self._shard_dir = self._cache_dir / "_shards"

    @property
    def path(self) -> str:
        return str(self._cache_dir)

    def exists(self) -> bool:
        """Check if cache has any .pt files."""
        return self._cache_dir.exists() and any(self._cache_dir.glob("*.pt"))

    def count(self) -> int:
        """Number of cached samples."""
        if not self._cache_dir.exists():
            return 0
        return len(list(self._cache_dir.glob("*.pt")))

    def indices(self) -> list[int]:
        """Sorted list of all cached sample indices."""
        if not self._cache_dir.exists():
            return []
        return sorted(int(f.stem) for f in self._cache_dir.glob("*.pt"))

    def save(self, idx: int, mel: torch.Tensor, tokens: torch.Tensor) -> None:
        """Save a single (mel, tokens) pair to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save((mel, tokens), self._cache_dir / f"{idx}.pt")

    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a single (mel, tokens) pair from disk."""
        mel, tokens = torch.load(
            self._cache_dir / f"{idx}.pt",
            weights_only=True, map_location="cpu",
        )
        mel_out = mel.contiguous().clone()
        tok_out = tokens.contiguous().clone()
        del mel, tokens
        return mel_out, tok_out

    def save_stream(self, stream: Iterator[tuple[torch.Tensor, torch.Tensor]],
                    source_label: str = "",
                    repo_id: str | None = None,
                    samples_per_shard: int = 50000) -> int:
        """Save an iterator of (mel, tokens) pairs to disk sequentially.

        When repo_id is set, uploads a shard to HF Hub every samples_per_shard
        samples and deletes the local .pt files. This makes caching resilient
        to container restarts — progress is persisted incrementally.

        Returns the number of samples saved.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        existing = self.indices()
        start_idx = (max(existing) + 1) if existing else 0
        count = 0
        shard_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Figure out next shard index from local + remote shards
        shard_idx = self._next_shard_idx(repo_id=repo_id)

        # Track which .pt indices belong to the current shard
        shard_start = start_idx

        for mel, tokens in stream:
            self.save(start_idx + count, mel, tokens)
            count += 1
            del mel, tokens

            if count % 1000 == 0:
                label = f"{source_label}: " if source_label else ""
                print(f"    {label}{count} samples cached ({start_idx + count} total)...",
                      flush=True)
                gc.collect()

            # Incremental upload when shard is full — read back from disk
            # instead of buffering in RAM to avoid OOM
            if repo_id and (count % samples_per_shard == 0):
                shard_end = start_idx + count
                self._upload_shard_from_disk(repo_id, shard_idx, shard_start, shard_end)
                self._delete_range(shard_start, shard_end)
                shard_start = shard_end
                shard_idx += 1
                gc.collect()

        # Upload remaining partial shard
        if repo_id and shard_start < start_idx + count:
            self._upload_shard_from_disk(repo_id, shard_idx, shard_start, start_idx + count)
            self._delete_range(shard_start, start_idx + count)
            shard_idx += 1

        gc.collect()
        return count

    def _next_shard_idx(self, repo_id: str | None = None) -> int:
        """Find the next shard index from local AND remote shard files.

        Checks both local _shards/ directory and the HF Hub repo to avoid
        overwriting shards that were uploaded before a container restart.
        """
        indices: list[int] = []

        # Local shards
        self._shard_dir.mkdir(parents=True, exist_ok=True)
        for f in self._shard_dir.glob("shard_*.pt"):
            try:
                indices.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass

        # Remote shards (survives container restarts)
        if repo_id:
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id, repo_type="dataset", token=_hf_token())
                for fname in files:
                    if fname.startswith("shard_") and fname.endswith(".pt"):
                        try:
                            indices.append(int(fname.split("_")[1].split(".")[0]))
                        except (IndexError, ValueError):
                            pass
            except Exception:
                pass

        return (max(indices) + 1) if indices else 0

    def _upload_shard(self, repo_id: str, shard_idx: int,
                      data: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Pack and upload a single shard to HF Hub."""
        from huggingface_hub import HfApi, create_repo

        token = _hf_token()
        api = HfApi(token=token)
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        except Exception:
            pass

        self._shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = self._shard_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(data, shard_path)
        size_mb = shard_path.stat().st_size / 1e6
        print(f"    Uploading shard_{shard_idx:04d}: {len(data)} samples ({size_mb:.0f} MB)...",
              flush=True)

        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"shard_{shard_idx:04d}.pt",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Shard {shard_idx}: {len(data)} samples",
        )
        # Delete local shard after upload to save disk
        shard_path.unlink(missing_ok=True)
        print(f"    Shard {shard_idx} uploaded and cleaned up.", flush=True)

    def _upload_shard_from_disk(self, repo_id: str, shard_idx: int,
                              start: int, end: int) -> None:
        """Read .pt files from disk, pack into a shard, and upload.

        Loads one file at a time to avoid holding the entire shard in RAM.
        """
        from huggingface_hub import HfApi, create_repo

        token = _hf_token()
        api = HfApi(token=token)
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        except Exception:
            pass

        # Collect samples from disk one at a time
        shard_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx in range(start, end):
            pt_path = self._cache_dir / f"{idx}.pt"
            if pt_path.exists():
                mel, tokens = torch.load(pt_path, weights_only=True, map_location="cpu")
                shard_data.append((mel, tokens))

        if not shard_data:
            return

        self._shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = self._shard_dir / f"shard_{shard_idx:04d}.pt"
        torch.save(shard_data, shard_path)
        size_mb = shard_path.stat().st_size / 1e6
        n = len(shard_data)
        del shard_data
        gc.collect()

        print(f"    Uploading shard_{shard_idx:04d}: {n} samples ({size_mb:.0f} MB)...",
              flush=True)

        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"shard_{shard_idx:04d}.pt",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Shard {shard_idx}: {n} samples",
        )
        shard_path.unlink(missing_ok=True)
        print(f"    Shard {shard_idx} uploaded and cleaned up.", flush=True)

    def _delete_range(self, start: int, end: int) -> None:
        """Delete .pt files in [start, end) to free disk after shard upload."""
        for idx in range(start, end):
            pt = self._cache_dir / f"{idx}.pt"
            pt.unlink(missing_ok=True)

    def upload_to_hub(self, repo_id: str, samples_per_shard: int = 50000) -> None:
        """Pack cached .pt files into shards and upload to a HF Dataset repo."""
        import json
        import shutil
        from huggingface_hub import HfApi, create_repo

        token = _hf_token()
        api = HfApi(token=token)
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
            print(f"  Dataset repo: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"  Repo creation note: {e}")

        pt_files = sorted(self._cache_dir.glob("*.pt"), key=lambda f: int(f.stem))
        total = len(pt_files)
        print(f"  Packing {total} samples into shards...")

        shard_dir = self._cache_dir / "_shards"
        shard_dir.mkdir(exist_ok=True)

        shard_idx = 0
        shard_data: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, pt_file in enumerate(pt_files):
            mel, tokens = torch.load(pt_file, weights_only=True, map_location="cpu")
            shard_data.append((mel, tokens))

            if len(shard_data) >= samples_per_shard or i == total - 1:
                shard_path = shard_dir / f"shard_{shard_idx:04d}.pt"
                torch.save(shard_data, shard_path)
                size_mb = shard_path.stat().st_size / 1e6
                print(f"    Shard {shard_idx}: {len(shard_data)} samples ({size_mb:.0f} MB)")
                shard_data = []
                shard_idx += 1

        metadata = {
            "total_samples": total,
            "num_shards": shard_idx,
            "samples_per_shard": samples_per_shard,
        }
        with open(shard_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Uploading {shard_idx} shards to {repo_id}...")
        api.upload_folder(
            folder_path=str(shard_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload mel cache: {total} samples in {shard_idx} shards",
        )
        print(f"  Upload complete!")
        shutil.rmtree(shard_dir)
