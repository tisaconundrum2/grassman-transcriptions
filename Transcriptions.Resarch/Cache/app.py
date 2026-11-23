#!/usr/bin/env python3
"""
Cache Mel Spectrograms — Pre-compute and upload to HF Hub.

No flags. Config via env vars. Just: python app.py
"""

import os
import shutil
import sys
import tempfile
import time
import threading

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))
sys.path.insert(0, _here)

from Shared.models.config import GrassmannConfig
from Shared.services.tokenizer_service import CharTokenizer
from Shared.repositories.dataset_repository import DatasetRepository
from Shared.repositories.cache_repository import CacheRepository
from Shared.services.dataset_service import DatasetService


# ── Config (env vars) ────────────────────────────────────────────────────

RECIPE = os.environ.get("RECIPE", "1750h")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", "tisaconundrum/grassmann-mel-cache")
INPUT_DIR = os.environ.get("INPUT_DIR", None)  # /mnt for mounted datasets
SAMPLES_PER_SHARD = int(os.environ.get("SAMPLES_PER_SHARD", "5000"))
WORKERS = int(os.environ.get("WORKERS", "0"))
MAX_PER_SOURCE = int(os.environ.get("MAX_PER_SOURCE", "0")) or None


def _start_health_server(port: int = 7860) -> None:
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Running...")

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"  Health server on port {port}", flush=True)


def _auto_pause_space() -> None:
    try:
        from huggingface_hub import HfApi
        space_id = os.environ.get("SPACE_ID")
        if space_id:
            HfApi().pause_space(space_id)
            print(f"  Space {space_id} paused.", flush=True)
    except Exception as e:
        print(f"  Note: could not auto-pause: {e}", flush=True)


def main() -> None:
    _start_health_server()

    mode = "mounted" if INPUT_DIR else "streaming"
    print("\n" + "=" * 60)
    print(f"  Grassmann — Dataset Caching ({mode})")
    print("=" * 60)
    print(f"  Recipe: {RECIPE}")
    print(f"  Output: {OUTPUT_REPO}")
    if INPUT_DIR:
        print(f"  Input:  {INPUT_DIR}")
    print("=" * 60 + "\n")

    config = GrassmannConfig()
    tokenizer = CharTokenizer()
    repo = DatasetRepository(config, tokenizer, input_dir=INPUT_DIR)
    staging_dir = tempfile.mkdtemp(prefix="mel_staging_")
    cache = CacheRepository(staging_dir)
    dataset = DatasetService(repo, cache)

    start = time.time()
    dataset.load_recipe(RECIPE,
                        max_samples_per_source=MAX_PER_SOURCE,
                        upload_repo=OUTPUT_REPO,
                        samples_per_shard=SAMPLES_PER_SHARD,
                        workers=WORKERS)
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"\n  Caching complete in {(time.time() - start)/60:.1f} minutes")

    _auto_pause_space()


if __name__ == "__main__":
    main()
