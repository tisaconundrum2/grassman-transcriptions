"""Create the Trackio Space and backfill per-epoch val_loss from checkpoint commit history.

Usage: backfill_trackio.py <run_name> <since ISO-8601 UTC>
Commit messages don't carry the run name, so `since` scopes commits to one run.
"""

import os
import re
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi

SPACE_ID = os.environ.get("TRACKIO_SPACE_ID", "tisaconundrum/grassmann-dashboard")
REPO = os.environ.get("CHECKPOINT_REPO", "tisaconundrum/grassmann-mel-cache")
RUN = sys.argv[1]
SINCE = datetime.fromisoformat(sys.argv[2]).replace(tzinfo=timezone.utc)

commits = HfApi().list_repo_commits(REPO, repo_type="dataset")
pat = re.compile(r"^Meta epoch (\d+): val_loss=([\d.]+)")
points = {}
for c in commits:
    m = pat.match(c.title or "")
    if m and c.created_at >= SINCE:
        points.setdefault(int(m.group(1)), (float(m.group(2)), c.created_at))

import trackio

trackio.init(project="grassmann", name=f"{RUN}-history", space_id=SPACE_ID,
             private=True, resume="allow", config={"source": "backfill", "run": RUN})
for epoch in sorted(points):
    val, when = points[epoch]
    trackio.log({"epoch": epoch, "epoch/val_loss": val}, step=epoch)
    print(f"  epoch {epoch:3d}  val_loss={val:.4f}  ({when:%Y-%m-%d %H:%M})")
trackio.finish()
print(f"Dashboard: https://huggingface.co/spaces/{SPACE_ID}")
