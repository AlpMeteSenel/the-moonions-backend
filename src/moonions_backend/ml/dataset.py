"""Automatically labelled training data for the hazard CNN.

Idea: the global LOLA map (474 m/px) is all we have for most of the Moon, but near the poles
LOLA has 80 m/px. We build pairs where
  * the INPUT is what the global map would show (80 m data block-averaged to ~474 m), and
  * the LABEL is the true fine-scale (80 m) safety of the 600 m landing zone.
The CNN thus learns to predict small-scale hazards that coarse maps can't resolve, and can
then be applied anywhere using LDEM_64.

Usage:  uv run python -m moonions_backend.ml.dataset --n 20000
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import CACHE
from ..dem import get_dem
from ..terrain import SAFE_SLOPE_DEG, analyze_zone, slope_grid

COARSE_RES_M = 473.8  # LDEM_64
PATCH = 32  # 32 x 474 m ≈ 15 km context window
FINE_FACTOR = 6  # 6 x 79 m fine samples per coarse pixel
CLASSES = ["safe", "marginal", "unsafe"]
OUT = CACHE / "hazard_dataset.npz"


def label_from_fine(t) -> int:
    if t.slope_p90_deg < SAFE_SLOPE_DEG and t.unsafe_fraction < 0.1:
        return 0
    if t.slope_deg > 12 or t.unsafe_fraction > 0.5:
        return 2
    return 1


def _channels(h: np.ndarray, res: float) -> np.ndarray:
    n = h.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(n * n)]
    coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
    detr = (h.ravel() - A @ coef).reshape(n, n)
    s = slope_grid(h, res)
    lap = np.gradient(np.gradient(h, res, axis=0), res, axis=0) + np.gradient(np.gradient(h, res, axis=1), res, axis=1)
    return np.stack([detr / 100.0, s / 20.0, np.clip(lap * 1000, -5, 5)]).astype(np.float32)


def patch_features(lat: float, lon: float, patch: int = PATCH, res_m: float = COARSE_RES_M) -> np.ndarray | None:
    """Model input at inference time: the global (coarse) DEM around a point."""
    dem = get_dem()
    src = dem.global_src
    half = res_m * (patch - 1) / 2
    _, _, h, _ = dem.local_grid(lat, lon, half, res_m, source=src)
    return _channels(h.astype(np.float64), res_m)


def coarse_from_fine(lat: float, lon: float, src) -> np.ndarray:
    """Simulate the coarse map from fine data: sample at 79 m then block-average 6x6."""
    dem = get_dem()
    fine_res = COARSE_RES_M / FINE_FACTOR
    n = PATCH * FINE_FACTOR
    half = fine_res * (n - 1) / 2
    _, _, h, _ = dem.local_grid(lat, lon, half, fine_res, source=src)
    h = h.reshape(PATCH, FINE_FACTOR, PATCH, FINE_FACTOR).mean(axis=(1, 3))
    return _channels(h.astype(np.float64), COARSE_RES_M)


def build(n: int, seed: int = 0) -> None:
    dem = get_dem()
    rng = np.random.default_rng(seed)
    X, y, meta = [], [], []
    for p in dem.polar:
        sign = -1 if p.south else 1
        k = n // max(1, len(dem.polar))
        # uniform over the cap area: sample cos-weighted colatitude
        colat = np.degrees(np.arccos(1 - rng.uniform(0, 1 - np.cos(np.radians(90 - p.min_abs_lat - 1.2)), k)))
        lats = sign * (90 - colat)
        lons = rng.uniform(-180, 180, k)
        for i, (la, lo) in enumerate(zip(lats, lons)):
            t = analyze_zone(float(la), float(lo), 300.0, source=p)
            X.append(coarse_from_fine(float(la), float(lo), p))
            y.append(label_from_fine(t))
            meta.append((la, lo))
            if i % 1000 == 0:
                print(p.name, i, "/", k, flush=True)
    X = np.stack(X)
    y = np.array(y, np.int64)
    np.savez_compressed(OUT, X=X, y=y, meta=np.array(meta, np.float32), classes=np.array(CLASSES))
    print("saved", OUT, X.shape, "class counts", np.bincount(y, minlength=3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    build(a.n)
