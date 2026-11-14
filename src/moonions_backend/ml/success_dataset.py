"""Dataset for the mission-success model (site x touchdown date → P(success)).

Inputs = planning-level NASA-derived features (80 m LOLA terrain patch + tabular features).
Labels = landing-mission simulator on LOLA 20 m terrain (see simulate.py).
Also stores two baselines evaluated on the same samples:
  * rule_score   : the dashboard's hand-weighted rule-based score (no backups/science)
  * phys80_td    : the same Monte-Carlo touchdown model run on the 80 m map (physics baseline)

  uv run python -m moonions_backend.ml.success_dataset --sites 8000
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool

import numpy as np

from ..config import CACHE
from ..craters import get_craters
from ..dem import get_dem
from ..horizon import Horizon
from ..terrain import analyze_zone
from .dataset import _channels
from .simulate import DEFAULT, get_truth_dem, time_outcomes, touchdown
from .sky import body_vectors_km, sky_many

OUT = CACHE / "success_dataset.npz"
PATCH = 32
PATCH_RES = 80.0
N_DATES = 48
DATES_PER_SITE = 4
STEPS = int(DEFAULT.mission_hours / DEFAULT.step_hours) + 1

TAB_FEATURES = [
    "slope_deg", "slope_p90_deg", "roughness_m", "unsafe_fraction", "relief_m", "phys80_td",
    "crater_density", "rim_km", "inside_crater",
    "sun_el0", "sun_clear0", "lit_frac", "longest_dark_d", "min_soc", "earth_clear0", "earth_frac",
    "colat",
]  # fmt: skip
HEADS = ["touchdown", "lit0", "energy_ok", "comms_ok"]


def date_list() -> list[datetime]:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [t0 + timedelta(days=22.83 * k, hours=(7 * k) % 24) for k in range(N_DATES)]


_VECS: list[dict] | None = None


def _vecs():
    global _VECS
    if _VECS is None:
        out = []
        for d in date_list():
            out.append(body_vectors_km([d + timedelta(hours=DEFAULT.step_hours * i) for i in range(STEPS)]))
        _VECS = out
    return _VECS


def terrain_patch80(lat: float, lon: float) -> np.ndarray:
    dem = get_dem()
    src = dem.source_for(lat)
    half = PATCH_RES * (PATCH - 1) / 2
    _, _, h, _ = dem.local_grid(lat, lon, half, PATCH_RES, source=src)
    return _channels(h.astype(np.float64), PATCH_RES)


def site_static(lat: float, lon: float) -> dict:
    """Date-independent planning features for a site."""
    dem = get_dem()
    src = dem.source_for(lat)
    t = analyze_zone(lat, lon, 300.0, source=src)
    phys = touchdown(lat, lon, DEFAULT, truth=src)["p_touchdown"]  # same MC model on the 80 m map
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
        "phys80_td": phys,
        "crater_density": c5["density_per_1000km2"],
        "rim_km": 5.0 if rim is None else min(rim, 5.0),
        "inside_crater": float(bool(c1.get("inside_crater"))),
        "colat": 90 - abs(lat),
        "elevation_m": t.elevation_m,
        "_terrain": t,
    }


def time_features(lat: float, lon: float, hz: Horizon, vecs: list[dict], elev_m: float = 0.0) -> dict[str, np.ndarray]:
    """Per-date time features for one site over a list of per-date body vectors → arrays (D,)."""
    sun_el, sun_az, e_el, e_az = [], [], [], []
    for v in vecs:
        s = sky_many([lat], [lon], v, height_m=[elev_m])
        sun_el.append(s["sun_el"][0])
        sun_az.append(s["sun_az"][0])
        e_el.append(s["earth_el"][0])
        e_az.append(s["earth_az"][0])
    sun_el, sun_az, e_el, e_az = map(np.array, (sun_el, sun_az, e_el, e_az))
    sun_clear = sun_el - hz.at(sun_az)
    earth_clear = e_el - hz.at(e_az)
    o = time_outcomes(sun_clear, sun_el, earth_clear, DEFAULT)
    o.update(
        {
            "sun_el0": sun_el[:, 0],
            "sun_clear0": sun_clear[:, 0],
            "earth_clear0": earth_clear[:, 0],
            "longest_dark_d": o["longest_dark_h"] / 24.0,
            "_sun_el": sun_el,
            "_sun_az": sun_az,
            "_earth_el": e_el,
            "_earth_az": e_az,
        }
    )
    return o


def rule_score(st: dict, tf: dict, i: int, hz: Horizon) -> float:
    """The dashboard's rule-based score (ground, sunlight, Earth, craters) for date index i."""
    from .. import scoring as sc

    steps = STEPS
    sky = {"sun_el": tf["_sun_el"][i], "sun_az": tf["_sun_az"][i], "earth_el": tf["_earth_el"][i], "earth_az": tf["_earth_az"][i]}
    f = {
        "ground": sc.ground_factor(st["_terrain"], None),
        "sunlight": sc.sunlight_factor(sky, hz, steps),
        "earth": sc.earth_factor(sky, hz, steps),
        "craters": sc.crater_factor(st["_lat"], st["_lon"]),
    }
    w = {"ground": 0.4, "sunlight": 0.25, "earth": 0.2, "craters": 0.15}
    score = sum(w[k] * f[k].score for k in w)
    if any(x.hard_fail for x in f.values()):
        score = min(score, 35.0)
    return float(score)


def _one(args):
    lat, lon, date_idx = args
    try:
        st = site_static(lat, lon)
        st["_lat"], st["_lon"] = lat, lon
        hz = Horizon(lat, lon)
        tr = touchdown(lat, lon, DEFAULT)
        vecs = _vecs()
        tf = time_features(lat, lon, hz, [vecs[i] for i in date_idx], st["elevation_m"])
        patch = terrain_patch80(lat, lon)
        rows = []
        for k, di in enumerate(date_idx):
            tab = [st[f] if f in st else float(tf[f][k]) for f in TAB_FEATURES]
            heads = [tr["p_touchdown"], tf["lit0"][k], tf["energy_ok"][k], tf["comms_ok"][k]]
            succ = float(np.prod(heads))
            rows.append((tab, heads, succ, rule_score(st, tf, k, hz), st["phys80_td"], di))
        return lat, lon, patch, rows, tr
    except Exception as exc:  # noqa: BLE001
        return None, None, None, str(exc), None


def build(n_sites: int, seed: int = 0, workers: int | None = None):
    assert get_truth_dem() is not None, "ldem_80s_20m.img missing: run scripts/download_data.sh"
    rng = np.random.default_rng(seed)
    # uniform over the cap area poleward of 80.6°S
    colat = np.degrees(np.arccos(1 - rng.uniform(0, 1 - np.cos(np.radians(9.4)), n_sites)))
    lats = -(90 - colat)
    lons = rng.uniform(-180, 180, n_sites)
    jobs = [(float(a), float(b), sorted(rng.choice(N_DATES, DATES_PER_SITE, replace=False).tolist())) for a, b in zip(lats, lons)]
    _vecs()  # compute ephemeris once before forking
    get_craters()
    X_patch, X_tab, Y, S, R, P, M = [], [], [], [], [], [], []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with Pool(workers) as pool:
        for i, (lat, lon, patch, rows, tr) in enumerate(pool.imap_unordered(_one, jobs, chunksize=16)):
            if lat is None:
                continue
            for tab, heads, succ, rule, phys, di in rows:
                X_patch.append(patch)
                X_tab.append(tab)
                Y.append(heads)
                S.append(succ)
                R.append(rule)
                P.append(phys)
                M.append((lat, lon, di))
            if i % 500 == 0:
                print(f"{i}/{n_sites}", flush=True)
    np.savez_compressed(
        OUT,
        X_patch=np.stack(X_patch).astype(np.float32),
        X_tab=np.array(X_tab, np.float32),
        Y=np.array(Y, np.float32),
        success=np.array(S, np.float32),
        rule=np.array(R, np.float32),
        phys80=np.array(P, np.float32),
        meta=np.array(M, np.float64),
        tab_features=np.array(TAB_FEATURES),
        heads=np.array(HEADS),
    )
    Y = np.array(Y)
    print("saved", OUT, len(S), "samples; mean heads", Y.mean(0).round(3), "mean success", np.mean(S).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    build(a.sites, workers=a.workers)
