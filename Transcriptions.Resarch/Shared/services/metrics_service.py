"""Trackio metrics logging. No-op unless TRACKIO_SPACE_ID is set."""

from __future__ import annotations

import os

_run = None
_step = 0


def init(project: str, name: str, config: dict) -> None:
    global _run
    space_id = os.environ.get("TRACKIO_SPACE_ID")
    if not space_id:
        return
    try:
        import trackio
        # capture_console keeps stdout/tracebacks with the run, since Azure keeps no logs.
        _run = trackio.init(project=project, name=name, config=config,
                            space_id=space_id, resume="allow", private=True,
                            capture_console=True)
        print(f"  Trackio: logging to https://huggingface.co/spaces/{space_id}", flush=True)
    except Exception as e:
        print(f"  Trackio disabled: {e}", flush=True)
        _run = None


def log(metrics: dict, step: int | None = None) -> None:
    global _step
    if _run is None:
        return
    _step = step if step is not None else _step + 1
    try:
        import trackio
        trackio.log(metrics, step=_step)
    except Exception as e:
        print(f"  Trackio log failed: {e}", flush=True)


def alert(title: str, text: str, level: str = "error") -> None:
    if _run is None:
        return
    try:
        import trackio
        lvl = getattr(trackio.alerts.AlertLevel, level.upper(), trackio.alerts.AlertLevel.ERROR)
        trackio.alert(title=title, text=text, level=lvl)
    except Exception as e:
        print(f"  Trackio alert failed: {e}", flush=True)


def finish() -> None:
    if _run is None:
        return
    try:
        import trackio
        trackio.finish()
    except Exception:
        pass
