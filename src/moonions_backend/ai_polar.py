"""AI landing-success map over the whole south-polar region, for any touchdown time.

Planning features for a 1.5 km grid are computed once (multiprocess) and cached; touchdown
safety is predicted once per model, and each requested time only needs the physics gates
(vectorised Sun/Earth geometry + horizons + battery). Output: an RGBA PNG in the same polar-stereographic
layout as the cap_s_* globe textures.

  uv run python -m moonions_backend.ai_polar --precompute
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import io
import os
from datetime import datetime, timedelta
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from .config import CACHE, MOON_RADIUS_M

log = logging.getLogger(__name__)

GRID = CACHE / "ai_polar_grid.npz"
SPACING_M = 1500.0
EDGE_LAT = 83.4  # map covers lat <= -83.4 (all Artemis III regions incl. Nobile Rim 2)
HALF_SIZE_M = 303960.0  # cap_s texture half-size (textures.json → caps.cap_s.half_size_m)
PNG_PX = 512


def _grid_xy():
    rho_max = 2 * MOON_RADIUS_M * np.tan(np.radians(90 - EDGE_LAT) / 2)
    ax = np.arange(-rho_max, rho_max + 1, SPACING_M)
    X, Y = np.meshgrid(ax, ax)
    m = np.hypot(X, Y) <= rho_max
    return X[m], Y[m]


def _xy_to_latlon(x, y):
    rho = np.hypot(x, y)
    colat = 2 * np.arctan(rho / (2 * MOON_RADIUS_M))
    return -(90 - np.degrees(colat)), np.degrees(np.arctan2(x, y))


def _static_one(args):
    lat, lon = args
    from .horizon import Horizon
    from .ml.td_dataset import TAB, patch80, planning_features

    try:
        pf = planning_features(lat, lon)
        return (np.array([pf[k] for k in TAB], np.float32), patch80(lat, lon), Horizon(lat, lon).profile.astype(np.float32), pf["elevation_m"])
    except Exception:  # noqa: BLE001
        return None


def precompute(workers: int | None = None):
    from .craters import get_craters

    x, y = _grid_xy()
    lat, lon = _xy_to_latlon(x, y)
    get_craters()
    rows = []
    with Pool(workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, r in enumerate(pool.imap(_static_one, list(zip(lat.tolist(), lon.tolist())), chunksize=64)):
            rows.append(r)
            if i % 5000 == 0:
                print(f"{i}/{len(lat)}", flush=True)
    ok = np.array([r is not None for r in rows])
    rows = [r for r in rows if r is not None]
    np.savez_compressed(
        GRID,
        x=x[ok], y=y[ok], lat=lat[ok], lon=lon[ok],
        tab=np.array([r[0] for r in rows], np.float32),
        patch=np.stack([r[1] for r in rows]).astype(np.float16),
        horizon=np.stack([r[2] for r in rows]),
        elev=np.array([r[3] for r in rows], np.float32),
    )  # fmt: skip
    print("saved", GRID, ok.sum(), "points")


@lru_cache(maxsize=1)
def _grid():
    return dict(np.load(GRID)) if GRID.exists() else None


def available() -> bool:
    return GRID.exists()


@lru_cache(maxsize=1)
def _touchdown():
    """Touchdown safety doesn't depend on time: predict the whole grid once (fast CNN ensemble)."""
    from . import ai

    g = _grid()
    if g is None or ai._load() is None:
        return None
    td, _ = ai.predict_td(g["tab"], g["patch"].astype(np.float32), g["lat"], g["lon"], variant=ai._fast_variant())
    return td.astype(np.float32)


@lru_cache(maxsize=48)
def components_at(when_iso: str) -> dict[str, np.ndarray] | None:
    """Per grid point: AI touchdown safety and the physics checks (light, power, Earth link)."""
    from . import ai

    g = _grid()
    td = _touchdown()
    if g is None or td is None:
        return None
    gates = ai.physics_gates(g["lat"], g["lon"], g["elev"], g["horizon"], [datetime.fromisoformat(when_iso)])
    return {"td": td, "lit": gates["lit0"][:, 0], "power": gates["energy_ok"][:, 0], "comms": gates["comms_ok"][:, 0]}


def success_at(when_iso: str) -> np.ndarray | None:
    c = components_at(when_iso)
    if c is None:
        return None
    return (c["td"] * c["lit"] * c["power"] * c["comms"]).astype(np.float32)


@lru_cache(maxsize=48)
def png(when_iso: str) -> bytes | None:
    from PIL import Image

    succ = success_at(when_iso)
    if succ is None:
        return None
    g = _grid()
    # rasterise the grid into the cap texture's pixel layout (row 0 = +y, col 0 = -x)
    px = HALF_SIZE_M * 2 / PNG_PX
    cols = np.clip(((g["x"] + HALF_SIZE_M) / px).astype(int), 0, PNG_PX - 1)
    rows = np.clip(((HALF_SIZE_M - g["y"]) / px).astype(int), 0, PNG_PX - 1)
    val = np.full((PNG_PX, PNG_PX), np.nan, np.float32)
    val[rows, cols] = succ
    # fill pinholes left by the 1.5 km grid vs ~1.19 km pixels
    from scipy.ndimage import uniform_filter

    valid = ~np.isnan(val)
    num = uniform_filter(np.where(valid, val, 0.0), 3)
    den = uniform_filter(valid.astype(np.float32), 3)
    val = np.where(~valid & (den > 0.2), num / np.maximum(den, 1e-6), val)
    # light NaN-aware smoothing: reads as areas instead of grid speckle
    from scipy.ndimage import gaussian_filter

    ok = ~np.isnan(val)
    sm = gaussian_filter(np.where(ok, val, 0.0), 0.9) / np.maximum(gaussian_filter(ok.astype(np.float32), 0.9), 1e-6)
    val = np.where(ok, sm, np.nan)
    # landing-success encoding shared with the frontend (palette.ts LAND_STOPS): nothing below
    # 30 % (the bare terrain means "no"), then one green that gets stronger and brighter
    stops = np.array([0.3, 0.5, 0.7, 0.85, 1.0], np.float32)
    cols = np.array([[6, 95, 70], [5, 150, 105], [16, 185, 129], [52, 211, 153], [167, 243, 208]], np.float32)
    alphas = np.array([0.0, 0.45, 0.75, 0.88, 0.95], np.float32)
    v = np.clip(np.nan_to_num(val, nan=0), 0, 1)
    rgb = np.stack([np.interp(v, stops, cols[:, k]) for k in range(3)], -1)
    alpha = np.where(np.isnan(val), 0, np.interp(v, stops, alphas) * 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.precompute:
        precompute(a.workers)


_warm_q: "queue.Queue[str] | None" = None
_warm_seen: set[str] = set()
WARM_HOURS = (1, -1, 2, -2, 3, 6, -6, 12, 24)


def _warm_worker() -> None:
    from . import diskcache

    assert _warm_q is not None
    while True:
        iso = _warm_q.get()
        try:
            diskcache.cached_bytes("polar_png", iso, lambda: png(iso))
        except Exception as exc:  # noqa: BLE001
            log.warning("polar map prewarm failed for %s: %s", iso, exc)


def prewarm(t: datetime) -> None:
    """Render the polar AI map for the hours around t in the background (one low-key thread)."""
    global _warm_q
    if _warm_q is None:
        _warm_q = queue.Queue()
        threading.Thread(target=_warm_worker, name="polar-prewarm", daemon=True).start()
    for h in WARM_HOURS:
        iso = (t + timedelta(hours=h)).isoformat()
        if iso not in _warm_seen:
            if len(_warm_seen) > 5000:
                _warm_seen.clear()
            _warm_seen.add(iso)
            _warm_q.put(iso)


def prewarm_one(t: datetime) -> None:
    """Queue a single hour (no neighbours)."""
    global _warm_q
    if _warm_q is None:
        _warm_q = queue.Queue()
        threading.Thread(target=_warm_worker, name="polar-prewarm", daemon=True).start()
    iso = t.isoformat()
    if iso not in _warm_seen:
        _warm_seen.add(iso)
        _warm_q.put(iso)


def region_summaries(regions: list[dict], when_iso: str) -> list[dict]:
    """Fast per-region viable area at a time, from the cached 1.5 km polar AI grid (for map colouring)."""
    from .dem import distance_bearing

    succ = success_at(when_iso)
    g = _grid()
    out = []
    for r in regions:
        half = float(r.get("half_size_m") or 7500.0)
        if succ is None or g is None or r["lat"] > -(EDGE_LAT - 0.2):
            out.append({"id": r.get("id"), "available": False})
            continue
        d, b = distance_bearing(r["lat"], r["lon"], g["lat"], g["lon"])
        e, n = d * np.sin(np.radians(b)), d * np.cos(np.radians(b))
        m = (np.abs(e) <= half) & (np.abs(n) <= half)
        if not m.any():
            out.append({"id": r.get("id"), "available": False})
            continue
        s = succ[m]
        c = components_at(when_iso)
        cell_km2 = (SPACING_M / 1000) ** 2
        shares = {
            "terrain": float((c["td"][m] >= 0.9).mean()),
            "sunlight": float(c["lit"][m].mean()),
            "power": float(c["power"][m].mean()),
            "earth_link": float(c["comms"][m].mean()),
        }
        # what limits landing here: the check that fails for the largest part of the region
        limiting = min(shares, key=shares.get)
        out.append(
            {
                "id": r.get("id"),
                "available": True,
                "viable_share": round(float((s >= 0.8).mean()), 4),
                "viable_km2": round(float((s >= 0.8).sum() * cell_km2), 1),
                "best_success": round(float(s.max()), 4),
                "shares": {k: round(v, 3) for k, v in shares.items()},
                "limiting": limiting,
                "samples": int(m.sum()),
            }
        )
    return out
