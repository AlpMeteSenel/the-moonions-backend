"""Small on-disk cache for slow, deterministic results (AI maps, region timelines).

Entries are keyed by the call arguments plus a version made from the trained model's checkpoint
mtime, so retraining invalidates them automatically. Survives restarts (uvicorn --reload).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import CACHE

log = logging.getLogger(__name__)
ROOT = CACHE / "results"
CODE_VERSION = "2"  # bump when the cached computations change shape or meaning


def _model_version() -> str:
    from .ai import CKPT

    try:
        return str(int(CKPT.stat().st_mtime))
    except OSError:
        return "none"


def _path(ns: str, key: Any, ext: str) -> Path:
    raw = json.dumps([CODE_VERSION, _model_version(), key], sort_keys=True, default=str)
    h = hashlib.sha1(raw.encode()).hexdigest()[:24]
    return ROOT / ns / f"{h}.{ext}"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)  # atomic: concurrent readers never see half a file


def cached_bytes(ns: str, key: Any, fn: Callable[[], bytes | None]) -> bytes | None:
    p = _path(ns, key, "bin")
    if p.exists():
        return p.read_bytes()
    data = fn()
    if data is not None:
        try:
            _write(p, data)
        except OSError as exc:
            log.warning("disk cache write failed: %s", exc)
    return data


def cached_json(ns: str, key: Any, fn: Callable[[], Any]) -> Any:
    p = _path(ns, key, "json")
    if p.exists():
        try:
            return json.loads(p.read_bytes())
        except ValueError:
            pass
    data = fn()
    if isinstance(data, dict) and data.get("available") is False:
        return data  # model not ready: don't pin that
    try:
        _write(p, json.dumps(data, separators=(",", ":")).encode())
    except OSError as exc:
        log.warning("disk cache write failed: %s", exc)
    return data
