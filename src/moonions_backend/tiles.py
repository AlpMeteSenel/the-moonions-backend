"""Quadtree height tiles of the LOLA polar caps, for level-of-detail rendering in the dashboard.

Tile space is the polar-stereographic square of the LOLA polar products (x east, y "up" on the
map, metres, [-H, H]^2 with H = half_size_m of the 80 m product). Level z has 2^z x 2^z tiles;
tile (z, tx, ty) has tx counted from -x and ty from +y (row 0 at the top, like the map image).

A tile is a (N+3) x (N+3) grid of raw LOLA DN values (int16, 0.5 m per unit, little-endian):
N = 128 segments, plus a one-sample border on every side so the client can compute seamless
normals. The finest levels read the 20 m south-pole product, coarser levels the 80 m product.
Tiles are cached on disk; they depend on nothing but the NASA source files.
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CACHE, RAW

log = logging.getLogger(__name__)
N = 128  # segments per tile edge
GRID = N + 3  # samples per tile edge incl. 1-sample border
TILE_CACHE = CACHE / "tiles"


@lru_cache(maxsize=4)
def _source(pole: str, res: int):
    from .dem import PolarImg

    img, lbl = RAW / f"ldem_80{pole}_{res}m.img", RAW / f"ldem_80{pole}_{res}m.lbl"
    if not img.exists():
        return None
    return PolarImg(img, lbl)


def info() -> dict:
    """Tile pyramid description per pole (half-size, max level, sources)."""
    out = {}
    for pole in ("s", "n"):
        base = _source(pole, 80)
        if base is None:
            continue
        half = base.sample_off * base.map_scale
        fine = _source(pole, 20)
        finest = fine.map_scale if fine else base.map_scale
        # deepest level whose sample spacing is still >= ~0.9x the finest source resolution
        max_z = int(np.floor(np.log2(2 * half / (N * finest * 0.9))))
        out[pole] = {"half_size_m": half, "segments": N, "max_level": max_z, "finest_m": finest, "sources": [f"{s.name}" for s in (base, fine) if s]}
    return out


def _sample(src, xs: np.ndarray, ys: np.ndarray, spacing: float) -> np.ndarray:
    """Bilinear DN at projection coords (xs along columns, ys along rows), box-prefiltered by stride."""
    cols = src.sample_off + xs / src.map_scale
    rows = src.line_off - ys / src.map_scale
    k = max(1, int(spacing / src.map_scale))  # read every k-th source sample (cheap prefilter)
    r0 = int(np.clip(np.floor(rows.min()) - k, 0, src.lines - 1))
    r1 = int(np.clip(np.ceil(rows.max()) + k + 1, 1, src.lines))
    c0 = int(np.clip(np.floor(cols.min()) - k, 0, src.samples - 1))
    c1 = int(np.clip(np.ceil(cols.max()) + k + 1, 1, src.samples))
    win = np.asarray(src.arr[r0:r1:k, c0:c1:k], dtype=np.float32)
    rr = np.clip((rows - r0) / k, 0, win.shape[0] - 1.001)
    cc = np.clip((cols - c0) / k, 0, win.shape[1] - 1.001)
    i0, j0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
    fr, fc = rr - i0, cc - j0
    a = win[i0[:, None], j0[None, :]]
    b = win[i0[:, None], j0[None, :] + 1]
    c = win[i0[:, None] + 1, j0[None, :]]
    d = win[i0[:, None] + 1, j0[None, :] + 1]
    fc2, fr2 = fc[None, :], fr[:, None]
    return (a * (1 - fc2) + b * fc2) * (1 - fr2) + (c * (1 - fc2) + d * fc2) * fr2


def _build(pole: str, z: int, tx: int, ty: int) -> bytes | None:
    meta = info().get(pole)
    if meta is None or not (0 <= z <= meta["max_level"]) or not (0 <= tx < 2**z and 0 <= ty < 2**z):
        return None
    half = meta["half_size_m"]
    size = 2 * half / 2**z
    spacing = size / N
    x0 = -half + tx * size
    y0 = half - ty * size
    xs = x0 + (np.arange(GRID) - 1) * spacing
    ys = y0 - (np.arange(GRID) - 1) * spacing
    fine = _source(pole, 20)
    src = fine if (fine is not None and spacing < 60.0) else _source(pole, 80)
    dn = _sample(src, xs, ys, spacing)
    return np.clip(np.rint(dn), -32768, 32767).astype("<i2").tobytes()


def tile(pole: str, z: int, tx: int, ty: int) -> bytes | None:
    path: Path = TILE_CACHE / pole / str(z) / str(tx) / f"{ty}.bin"
    if path.exists():
        return path.read_bytes()
    data = _build(pole, z, tx, ty)
    if data is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("tile cache write failed: %s", exc)
    return data
