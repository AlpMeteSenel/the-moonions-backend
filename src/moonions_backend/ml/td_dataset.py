"""v2 dataset: touchdown safety per site (date-independent).

Mission success = P(safe touchdown) × light × power × Earth link. The last three are exact,
deterministic physics (ephemeris + terrain horizon + battery) and are computed, not learned. The
neural network learns the genuinely uncertain part: whether the lander touches down safely,
which depends on terrain finer than the planning maps show.

Per site we store
  label   : Monte-Carlo touchdown on LOLA 20 m terrain (simulate.touchdown, v2 physics)
  inputs  : 80 m LOLA patch, terrain statistics, crater stats, gravity deflection, latitude
  baselines: the same simulator run on the 80 m map (phys80) and the rule-based ground score
  horizon : terrain horizon profile + elevation, so light/power/comms can be evaluated for any date

  uv run python -m moonions_backend.ml.td_dataset --sites 16000
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np

from ..config import CACHE
from ..craters import get_craters
from ..dem import get_dem
from ..gravity import deflection_deg
from ..horizon import Horizon
from ..terrain import analyze_zone
from .dataset import _channels
from .simulate import DEFAULT, get_truth_dem, touchdown

OUT = CACHE / "td_dataset.npz"
PATCH = 32
PATCH_RES = 80.0
TAB = [
    "slope_deg", "slope_p90_deg", "roughness_m", "unsafe_fraction", "relief_m",
    "phys80_td", "phys80_direct", "phys80_tip",
    "crater_density", "rim_km", "inside_crater", "deflection_deg", "colat",
]  # fmt: skip


def patch80(lat: float, lon: float) -> np.ndarray:
    dem = get_dem()
    src = dem.source_for(lat)
    half = PATCH_RES * (PATCH - 1) / 2
    _, _, h, _ = dem.local_grid(lat, lon, half, PATCH_RES, source=src)
    return _channels(h.astype(np.float64), PATCH_RES)


def planning_features(lat: float, lon: float) -> dict:
    """Everything a planner can compute from the 80 m maps (no 20 m truth)."""
    dem = get_dem()
    src = dem.source_for(lat)
    t = analyze_zone(lat, lon, 300.0, source=src)
    ph = touchdown(lat, lon, DEFAULT, truth=src)
    db = get_craters()
    c5 = db.around(lat, lon, 5000) if db else {"density_per_1000km2": 0.0}
    c1 = db.around(lat, lon, 1500) if db else {"nearest_rim_km": None, "inside_crater": False}
    rim = c1.get("nearest_rim_km")
    return {
        "slope_deg": t.slope_deg,
        "slope_p90_deg": t.slope_p90_deg,
        "roughness_m": t.roughness_m,
        "unsafe_fraction": t.unsafe_fraction,
        "relief_m": t.relief_m,
        "phys80_td": ph["p_touchdown"],
        "phys80_direct": ph["p_direct"],
        "phys80_tip": ph["p_tipover"],
        "crater_density": c5["density_per_1000km2"],
        "rim_km": 5.0 if rim is None else min(rim, 5.0),
        "inside_crater": float(bool(c1.get("inside_crater"))),
        "deflection_deg": float(deflection_deg(lat, lon)),
        "colat": 90 - abs(lat),
        "elevation_m": t.elevation_m,
        "_terrain": t,
    }


def _one(args):
    lat, lon = args
    try:
        from ..scoring import ground_factor

        pf = planning_features(lat, lon)
        tr = touchdown(lat, lon, DEFAULT)
        return (
            lat,
            lon,
            np.array([pf[k] for k in TAB], np.float32),
            patch80(lat, lon),
            np.array([tr["p_touchdown"], tr["p_direct"], tr["p_tipover"], tr["p_gear"]], np.float32),
            float(ground_factor(pf["_terrain"], None).score),
            Horizon(lat, lon).profile.astype(np.float32),
            pf["elevation_m"],
        )
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def build(n_sites: int, seed: int = 1, workers: int | None = None):
    assert get_truth_dem() is not None, "ldem_80s_20m.img missing"
    rng = np.random.default_rng(seed)
    colat = np.degrees(np.arccos(1 - rng.uniform(0, 1 - np.cos(np.radians(9.4)), n_sites)))
    lats, lons = -(90 - colat), rng.uniform(-180, 180, n_sites)
    get_craters()
    rows, errors = [], 0
    with Pool(workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        for i, r in enumerate(pool.imap(_one, list(zip(lats.tolist(), lons.tolist())), chunksize=32)):
            if isinstance(r, str):
                errors += 1
                continue
            rows.append(r)
            if i % 2000 == 0:
                print(f"{i}/{n_sites}", flush=True)
    lat, lon, tab, patch, y, rule, hz, elev = map(np.array, zip(*rows))
    np.savez_compressed(
        OUT, lat=lat, lon=lon, X_tab=tab.astype(np.float32), X_patch=patch.astype(np.float32), Y=y.astype(np.float32),
        rule=rule.astype(np.float32), horizon=hz.astype(np.float32), elev=elev.astype(np.float32), tab_features=np.array(TAB),
        targets=np.array(["p_touchdown", "p_direct", "p_tipover", "p_ops_limit"]),
    )  # fmt: skip
    print(f"saved {OUT}: {len(lat)} sites ({errors} failed); touchdown mean {y[:,0].mean():.3f}, <0.5: {(y[:,0]<0.5).mean():.3f}, ==1: {(y[:,0]==1).mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    build(a.sites, workers=a.workers)
