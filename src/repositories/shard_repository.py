"""
Shard Repositories — interchangeable strategies for fetching mel shards.

  LocalShardRepository     read from a local dir or HF Jobs mounted volume
                           (-v hf://datasets/<repo>:/data:ro)
  HubShardRepository       download one shard at a time from the Hub
  PrefetchShardRepository  Hub download with shard n+1 fetched while n trains

Use make_shard_repository() to pick one from a source string and mode.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import torch

from src.interfaces.i_shard_repository import IShardRepository, Sample

_EST_SAMPLES_PER_SHARD = 20000


def _is_shard(name: str) -> bool:
    return name.startswith("shard_") and name.endswith(".pt")


def _load_shard(path: str | Path) -> list[Sample]:
    return torch.load(path, weights_only=True, map_location="cpu")


class LocalShardRepository(IShardRepository):
    """Shards on a local filesystem — including HF Jobs mounted volumes.

    Mounts are lazy: each file is fetched on first read and cached on the
    Job's ephemeral disk, so there's no download step here.
    """

    def __init__(self, path: str):
        self._base = Path(path)

    def list_shards(self) -> list[str]:
        return sorted(f.name for f in self._base.iterdir() if _is_shard(f.name))

    def total_samples(self) -> int:
        meta = self._base / "metadata.json"
        if meta.is_file():
            with open(meta) as f:
                return json.load(f).get("total_samples", 0) or \
                    len(self.list_shards()) * _EST_SAMPLES_PER_SHARD
        return len(self.list_shards()) * _EST_SAMPLES_PER_SHARD

    def iter_shards(self, shard_names: list[str]) -> Iterator[list[Sample]]:
        for name in shard_names:
            data = _load_shard(self._base / name)
            yield data
            del data
            gc.collect()


class HubShardRepository(IShardRepository):
    """Downloads shards from a HF Dataset repo, one at a time, sequentially.

    Each shard goes to its own tempdir, which is removed right after loading,
    so disk usage stays at one shard.
    """

    def __init__(self, repo_id: str, token: str | None = None):
        self._repo_id = repo_id
        self._token = token or os.environ.get("HF_TOKEN")

    def list_shards(self) -> list[str]:
        from huggingface_hub import list_repo_files
        files = list_repo_files(self._repo_id, repo_type="dataset", token=self._token)
        return sorted(f for f in files if _is_shard(f))

    def total_samples(self) -> int:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self._repo_id, "metadata.json",
                                   repo_type="dataset", token=self._token)
            with open(path) as f:
                total = json.load(f).get("total_samples", 0)
            if total:
                return total
        except Exception:
            pass
        return len(self.list_shards()) * _EST_SAMPLES_PER_SHARD

    def _download(self, shard_name: str) -> tuple[str, str]:
        """Download a shard file to a fresh tempdir. Returns (file_path, tmp_dir)."""
        from huggingface_hub import hf_hub_download
        tmp_dir = tempfile.mkdtemp(prefix="hf_shard_")
        path = hf_hub_download(self._repo_id, shard_name, repo_type="dataset",
                               token=self._token, cache_dir=tmp_dir)
        return path, tmp_dir

    @staticmethod
    def _load_and_clean(path: str, tmp_dir: str) -> list[Sample]:
        try:
            return _load_shard(path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def iter_shards(self, shard_names: list[str]) -> Iterator[list[Sample]]:
        for name in shard_names:
            data = self._load_and_clean(*self._download(name))
            yield data
            del data
            gc.collect()


class PrefetchShardRepository(HubShardRepository):
    """Hub download with one-ahead prefetch.

    A background thread downloads shard n+1 (file only) while shard n is
    being consumed. torch.load is deferred until shard n is released, so
    RAM holds one shard; disk holds at most one pending download.
    """

    def iter_shards(self, shard_names: list[str]) -> Iterator[list[Sample]]:
        if not shard_names:
            return
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            pending: Future = executor.submit(self._download, shard_names[0])
            for i in range(len(shard_names)):
                path, tmp_dir = pending.result()
                if i + 1 < len(shard_names):
                    pending = executor.submit(self._download, shard_names[i + 1])
                data = self._load_and_clean(path, tmp_dir)
                yield data
                del data
                gc.collect()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def _is_tar_shard(name: str) -> bool:
    return name.startswith("shard_") and name.endswith(".tar")


def _load_tar_shard(path: str | Path) -> list[Sample]:
    """Load a .tar shard produced by the .NET Precalculate.Mel.Spectrograms worker.

    Each sample is two consecutive entries: NNNNNN.f32 (little-endian float32 mel)
    and NNNNNN.json (metadata with shape, token_ids). Entries are always written
    in order by the .NET writer, so we stream them in a single pass without
    buffering the entire tar into a dict.
    """
    import tarfile
    import numpy as np

    result: list[Sample] = []
    pending_mel: bytes | None = None
    pending_key: str | None = None

    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, ext = member.name.rsplit(".", 1)
            f = tar.extractfile(member)
            if f is None:
                continue

            if ext == "f32":
                pending_mel = f.read()
                pending_key = key
            elif ext == "json" and pending_mel is not None and pending_key == key:
                meta = json.loads(f.read())
                shape = meta["shape"]  # [mel_bins, frames]
                mel = torch.from_numpy(
                    np.frombuffer(pending_mel, dtype="<f4").copy().reshape(shape)
                )
                tokens = torch.tensor(meta["token_ids"], dtype=torch.long)
                result.append((mel, tokens))
                pending_mel = None
                pending_key = None

    return result


class TarShardRepository(IShardRepository):
    """Reads .tar shards produced by the .NET mel spectrogram worker.

    Works with both local paths and HF Hub repos. Tar shards contain
    paired .f32 (mel data) and .json (metadata) entries per sample.
    """

    def __init__(self, path: str):
        self._base = Path(path)

    def list_shards(self) -> list[str]:
        return sorted(f.name for f in self._base.iterdir() if _is_tar_shard(f.name))

    def total_samples(self) -> int:
        return len(self.list_shards()) * 5000  # Default shard size

    def iter_shards(self, shard_names: list[str]) -> Iterator[list[Sample]]:
        for name in shard_names:
            data = _load_tar_shard(self._base / name)
            yield data
            del data
            gc.collect()


class HubTarShardRepository(HubShardRepository):
    """Downloads .tar shards from HF Hub, one at a time.

    Same download logic as HubShardRepository but loads .tar instead of .pt.
    """

    def list_shards(self) -> list[str]:
        from huggingface_hub import list_repo_files
        files = list_repo_files(self._repo_id, repo_type="dataset", token=self._token)
        # Tar shards are under shards/ prefix
        return sorted(
            f for f in files
            if f.endswith(".tar") and "shard_" in f
        )

    @staticmethod
    def _load_and_clean(path: str, tmp_dir: str) -> list[Sample]:
        try:
            return _load_tar_shard(path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class PrefetchTarShardRepository(HubTarShardRepository):
    """Hub download of .tar shards with one-ahead prefetch.

    Same prefetch strategy as PrefetchShardRepository: a background thread
    downloads shard n+1 while shard n is being consumed.
    """

    def iter_shards(self, shard_names: list[str]) -> Iterator[list[Sample]]:
        if not shard_names:
            return
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            pending: Future = executor.submit(self._download, shard_names[0])
            for i in range(len(shard_names)):
                path, tmp_dir = pending.result()
                if i + 1 < len(shard_names):
                    pending = executor.submit(self._download, shard_names[i + 1])
                data = self._load_and_clean(path, tmp_dir)
                yield data
                del data
                gc.collect()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


FETCH_MODES = ("auto", "local", "stream", "prefetch")


def make_shard_repository(source: str, mode: str = "auto") -> IShardRepository:
    """Pick a shard repository.

    auto      local if `source` is an existing directory, else prefetch.
              Auto-detects .tar vs .pt shards.
    local     LocalShardRepository for .pt, TarShardRepository for .tar
    stream    HubShardRepository for .pt, HubTarShardRepository for .tar
    prefetch  PrefetchShardRepository for .pt (tar uses stream for now)
    """
    if mode == "auto":
        mode = "local" if os.path.isdir(source) else "prefetch"

    if mode == "local":
        path = Path(source)
        # Auto-detect format: check for .tar shards first, then .pt
        if path.is_dir():
            has_tar = any(f.name.endswith(".tar") for f in path.iterdir() if "shard_" in f.name)
            # Also check shards/ subdirectory
            shards_dir = path / "shards"
            if shards_dir.is_dir():
                has_tar = any(f.name.endswith(".tar") for f in shards_dir.iterdir() if "shard_" in f.name)
                if has_tar:
                    return TarShardRepository(str(shards_dir))
            if has_tar:
                return TarShardRepository(source)
        return LocalShardRepository(source)

    if mode == "stream":
        # Check if repo has .tar shards
        try:
            from huggingface_hub import list_repo_files
            token = os.environ.get("HF_TOKEN")
            files = list_repo_files(source, repo_type="dataset", token=token)
            has_tar = any(f.endswith(".tar") and "shard_" in f for f in files)
            if has_tar:
                return HubTarShardRepository(source)
        except Exception:
            pass
        return HubShardRepository(source)

    if mode == "prefetch":
        # Check for .tar shards in repo
        try:
            from huggingface_hub import list_repo_files
            token = os.environ.get("HF_TOKEN")
            files = list_repo_files(source, repo_type="dataset", token=token)
            has_tar = any(f.endswith(".tar") and "shard_" in f for f in files)
            if has_tar:
                return PrefetchTarShardRepository(source)
        except Exception:
            pass
        return PrefetchShardRepository(source)

    raise ValueError(f"Unknown fetch mode '{mode}', expected one of {FETCH_MODES}")
