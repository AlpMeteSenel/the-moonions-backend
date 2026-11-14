"""Region-level landing analysis.

NASA plans landings as *regions* (e.g. the 15 x 15 km Artemis III candidate regions) and then targets
a *landing ellipse* inside them. For a region and a touchdown time we compute:

* an AI + physics success map on a regular grid (default 250 m): AI touchdown safety × light ×
  power × Earth link for every cell;
* the safe landable area and the share of the region that is viable at that time;
* **landing zones**: connected patches of high success big enough to contain the landing ellipse
  (3σ navigation scatter), with the ellipse placed at the most interior point of each zone;
* over a period: the viable share of the region at each time → region landing windows.

Planning features per cell are the same as the touchdown model's training features. Terrain
horizons vary slowly, so they are computed on a 1 km sub-grid and interpolated.
"""

from __future__ import annotations

import hashlib
import os
import atexit
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
from scipy import ndimage

from .config import CACHE
from .dem import offset_latlon
from .ephemeris import to_utc

REGION_CACHE = CACHE / "regions"
REGION_CACHE.mkdir(parents=True, exist_ok=True)
ZONE_SUCCESS = 0.8  # a cell counts toward a landing zone above this predicted success
SAFE_TD = 0.9  # "safe ground" for area statistics


def _feat(args):
    lat, lon = args
    from .ml.td_dataset import TAB, patch80, planning_features

    try:
        pf = planning_features(lat, lon)
        return np.array([pf[k] for k in TAB], np.float32), patch80(lat, lon).astype(np.float32), float(pf["elevation_m"])
    except Exception:  # noqa: BLE001
        return None


def _hz(args):
    from .horizon import Horizon

    lat, lon = args
    return Horizon(lat, lon).profile.astype(np.float32)


_POOL: ProcessPoolExecutor | None = None
POOL_WORKERS = int(os.environ.get("MOONIONS_POOL_WORKERS", "4"))  # each worker holds its own DEM pages


def _pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=max(1, min(POOL_WORKERS, (os.cpu_count() or 2) - 1)))
        atexit.register(shutdown_pool)
    return _POOL


def shutdown_pool() -> None:
    """Stop the horizon workers (called on server shutdown/reload so they never outlive it)."""
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
        for proc in list(getattr(_POOL, "_processes", {}).values()):
            proc.terminate()
        _POOL = None


def _key(lat, lon, half, cells):
    return hashlib.sha1(f"{lat:.6f},{lon:.6f},{half:.1f},{cells},v2".encode()).hexdigest()[:16]


@lru_cache(maxsize=32)
def region_grid(lat: float, lon: float, half_size_m: float, cells: int) -> dict:
    """Planning features + horizons for a cells x cells grid over the region (cached on disk)."""
    path = REGION_CACHE / f"{_key(lat, lon, half_size_m, cells)}.npz"
    if path.exists():
        return dict(np.load(path))
    axis = np.linspace(-half_size_m, half_size_m, cells)
    E, N = np.meshgrid(axis, axis[::-1])  # row 0 = north
    glat, glon = offset_latlon(lat, lon, E.ravel(), N.ravel())
    pts = list(zip(glat.tolist(), glon.tolist()))
    feats = list(_pool().map(_feat, pts, chunksize=32))
    ok = np.array([f is not None for f in feats])
    if not ok.any():
        raise ValueError("No terrain data for this region")
    tab = np.stack([f[0] if f is not None else np.zeros_like(feats[int(np.argmax(ok))][0]) for f in feats])
    patch = np.stack([f[1] if f is not None else np.zeros_like(feats[int(np.argmax(ok))][1]) for f in feats])
    elev = np.array([f[2] if f is not None else 0.0 for f in feats], np.float32)
    # horizons on a coarse sub-grid (≈1 km), bilinearly interpolated to every cell
    hc = max(3, int(round(2 * half_size_m / 1000)) + 1)
    hax = np.linspace(-half_size_m, half_size_m, hc)
    HE, HN = np.meshgrid(hax, hax[::-1])
    hlat, hlon = offset_latlon(lat, lon, HE.ravel(), HN.ravel())
    prof = np.stack(list(_pool().map(_hz, list(zip(hlat.tolist(), hlon.tolist())), chunksize=8))).reshape(hc, hc, -1)
    fx = (E.ravel() + half_size_m) / (2 * half_size_m) * (hc - 1)
    fy = (half_size_m - N.ravel()) / (2 * half_size_m) * (hc - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, hc - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, hc - 2)
    ax_, ay_ = (fx - x0)[:, None], (fy - y0)[:, None]
    horizon = (prof[y0, x0] * (1 - ax_) + prof[y0, x0 + 1] * ax_) * (1 - ay_) + (prof[y0 + 1, x0] * (1 - ax_) + prof[y0 + 1, x0 + 1] * ax_) * ay_
    out = {"lat": glat, "lon": glon, "east": E.ravel(), "north": N.ravel(), "tab": tab, "patch": patch.astype(np.float16), "elev": elev, "horizon": horizon.astype(np.float32), "ok": ok}
    np.savez_compressed(path, **out)
    return out


@lru_cache(maxsize=32)
def region_touchdown(lat: float, lon: float, half_size_m: float, cells: int):
    from . import ai

    g = region_grid(lat, lon, half_size_m, cells)
    td, sd = ai.predict_td(g["tab"], g["patch"].astype(np.float32), g["lat"], g["lon"], variant=ai._fast_variant())
    td = np.where(g["ok"], td, 0.0)
    return td.astype(np.float32), sd.astype(np.float32)


def _zones(success: np.ndarray, cells: int, res_m: float, ellipse_m: float, east, north, lat, lon, max_zones: int = 6) -> list[dict]:
    S = success.reshape(cells, cells)
    good = S >= ZONE_SUCCESS
    lab, n = ndimage.label(good)
    if n == 0:
        return []
    # EDT measures centre-to-centre distance, not clearance to the unsafe cell's
    # boundary. Subtract its half-diagonal for a conservative clearance bound.
    # Pad with unsafe cells so the ellipse also stays inside the analysed region.
    # Without this, a lone 250 m cell falsely claims room for a 180 m radius.
    padded = np.pad(good, 1, constant_values=False)
    margin = np.maximum(0.0, ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * res_m - res_m / np.sqrt(2))
    # The first and last samples lie on the region boundary, not half a cell
    # inside it. Clamp clearance to that exact boundary as well.
    row, col = np.indices(good.shape)
    boundary = np.minimum.reduce([row, col, cells - 1 - row, cells - 1 - col]) * res_m
    margin = np.minimum(margin, boundary)
    zones = []
    for z in range(1, n + 1):
        m = lab == z
        area_km2 = m.sum() * (res_m / 1000) ** 2
        mz = np.where(m, margin, -1)
        k = int(np.argmax(mz))
        if mz.flat[k] < ellipse_m:
            continue  # the landing ellipse doesn't fit inside this patch
        zones.append(
            {
                "lat": float(lat[k]),
                "lon": float(lon[k]),
                "east_m": float(east[k]),
                "north_m": float(north[k]),
                "area_km2": round(float(area_km2), 3),
                "margin_m": round(float(mz.flat[k]), 0),
                "mean_success": round(float(S[m].mean()), 4),
                "target_success": round(float(S.flat[k]), 4),
                "fits_ellipse": bool(mz.flat[k] >= ellipse_m),
            }
        )
    zones.sort(key=lambda z: (-(z["fits_ellipse"]), -z["mean_success"] * np.sqrt(z["area_km2"])))
    for i, z in enumerate(zones[:max_zones], 1):
        z["rank"] = i
        z["id"] = f"Z{i}"
    return zones[:max_zones]


def analyze(lat: float, lon: float, half_size_m: float, when: datetime, cells: int = 61) -> dict:
    from . import ai
    from .backups import compass, divert_cost
    from .dem import distance_bearing
    from .ml.simulate import DEFAULT

    if ai._load() is None:
        return {"available": False}
    g = region_grid(round(lat, 6), round(lon, 6), float(half_size_m), int(cells))
    td, sd = region_touchdown(round(lat, 6), round(lon, 6), float(half_size_m), int(cells))
    gates = ai.physics_gates(g["lat"], g["lon"], g["elev"], g["horizon"], [when])
    gate = (gates["lit0"] * gates["energy_ok"] * gates["comms_ok"])[:, 0]
    success = td * gate
    res = 2 * half_size_m / (cells - 1)
    ellipse = 3 * DEFAULT.nav_sigma_m
    zones = _zones(success, cells, res, ellipse, g["east"], g["north"], g["lat"], g["lon"])
    if zones:
        z0 = zones[0]
        for z in zones[1:]:
            d, b = distance_bearing(z0["lat"], z0["lon"], z["lat"], z["lon"])
            z["from_primary"] = {"distance_m": round(float(d), 0), "direction": compass(float(b)), "decide_by_s": divert_cost(float(d))["decide_by_s"]}
    cell_km2 = (res / 1000) ** 2
    return {
        "available": True,
        "lat": lat,
        "lon": lon,
        "half_size_m": half_size_m,
        "cells": cells,
        "resolution_m": res,
        "landing_time": to_utc(when).isoformat(),
        "ellipse_radius_m": ellipse,
        "success": np.round(success, 3).tolist(),
        "touchdown": np.round(td, 3).tolist(),
        "uncertainty": np.round(sd, 3).tolist(),
        "gate": gate.astype(int).tolist(),
        "stats": {
            "area_km2": round((2 * half_size_m / 1000) ** 2, 1),
            "safe_ground_km2": round(float((td >= SAFE_TD).sum() * cell_km2), 2),
            "safe_ground_share": round(float((td >= SAFE_TD).mean()), 4),
            "lit_share": round(float(gates["lit0"][:, 0].mean()), 4),
            "comms_share": round(float(gates["comms_ok"][:, 0].mean()), 4),
            "power_share": round(float(gates["energy_ok"][:, 0].mean()), 4),
            "viable_share": round(float((success >= ZONE_SUCCESS).mean()), 4),
            "viable_km2": round(float((success >= ZONE_SUCCESS).sum() * cell_km2), 2),
            "best_success": round(float(success.max()), 4),
            "zones": len(zones),
        },
        "zones": zones,
    }


def timeline(lat: float, lon: float, half_size_m: float, start: datetime, days: float = 30, step_hours: float = 6, cells: int = 41) -> dict:
    """Viable share of the region and best-zone success over time → region landing windows."""
    from . import ai

    if ai._load() is None:
        return {"available": False}
    g = region_grid(round(lat, 6), round(lon, 6), float(half_size_m), int(cells))
    td, _ = region_touchdown(round(lat, 6), round(lon, 6), float(half_size_m), int(cells))
    times = [to_utc(start) + timedelta(hours=step_hours * k) for k in range(int(days * 24 / step_hours) + 1)]
    gates = ai.physics_gates(g["lat"], g["lon"], g["elev"], g["horizon"], times)
    gate = gates["lit0"] * gates["energy_ok"] * gates["comms_ok"]  # (cells², T)
    succ = td[:, None] * gate
    viable = (succ >= ZONE_SUCCESS).mean(0)
    # best landing spot each time: the 95th-percentile cell (robust to single-cell noise)
    best = np.percentile(succ, 99, axis=0)
    score = np.sqrt(np.clip(viable, 0, 1)) * best  # rewards both a good spot and room to manoeuvre
    return {
        "available": True,
        "times": [t.isoformat() for t in times],
        "viable_share": np.round(viable, 4).tolist(),
        "best_success": np.round(best, 4).tolist(),
        "score": np.round(score, 4).tolist(),
        "windows": ai.windows(times, best * (viable > 0.02), step_hours, threshold=ZONE_SUCCESS),
        "step_hours": step_hours,
    }


def recommend(regions: list[dict], start: datetime, days: float, step_hours: float, cells: int = 41) -> dict:
    """Pick the best region and touchdown time; landing zones inside it are the primary target and backups."""
    from . import ai

    if ai._load() is None:
        return {"available": False}
    tls = [timeline(r["lat"], r["lon"], r.get("half_size_m", 7500.0), start, days, step_hours, cells) for r in regions]
    times = [datetime.fromisoformat(t) for t in tls[0]["times"]]
    score = np.array([t["score"] for t in tls])
    best = np.array([t["best_success"] for t in tls])
    viable = np.array([t["viable_share"] for t in tls])
    i, j = np.unravel_index(np.argmax(score), score.shape)
    reg = regions[i]
    detail = analyze(reg["lat"], reg["lon"], reg.get("half_size_m", 7500.0), times[j], cells=61)
    zones = detail["zones"]
    region_windows = tls[i]["windows"]
    for w in region_windows:
        w["is_primary"] = abs((datetime.fromisoformat(w["best"]) - times[j]).total_seconds()) < step_hours * 3600 * 0.5 or (
            datetime.fromisoformat(w["start"]) <= times[j] < datetime.fromisoformat(w["end"])
        )
    region_best = sorted(
        (
            {
                "site_id": regions[a].get("id"),
                "name": regions[a].get("name"),
                "landing_time": times[int(np.argmax(score[a]))].isoformat(),
                "success_probability": round(float(best[a, int(np.argmax(score[a]))]), 4),
                "viable_share": round(float(viable[a, int(np.argmax(score[a]))]), 4),
                "open_slots": int((best[a] >= ZONE_SUCCESS).sum()),
            }
            for a in range(len(regions))
        ),
        key=lambda x: -x["success_probability"] * np.sqrt(max(x["viable_share"], 0)),
    )
    st = detail["stats"]
    expl = [
        f"{reg.get('name')} on {times[j].strftime('%d %b %H:%M UTC')}: {st['viable_km2']} km² of the {st['area_km2']:g} km² region is viable "
        f"({st['viable_share']*100:.0f}%), best landing spot {st['best_success']*100:.0f}%."
    ]
    if zones:
        z = zones[0]
        expl.append(f"Primary landing zone Z1: {z['area_km2']} km², {z['margin_m']:.0f} m clearance to the zone boundary, {z['mean_success']*100:.0f}% average success.")
        if len(zones) > 1:
            expl.append(f"{len(zones)-1} backup zones inside the region, nearest {min(x['from_primary']['distance_m'] for x in zones[1:]) / 1000:.1f} km away.")
    else:
        expl.append("No zone in this region is large enough for the landing ellipse at this time.")
    slips = [w for w in region_windows if not w.get("is_primary")]
    if slips:
        expl.append("If the launch slips, the region reopens " + ", ".join(datetime.fromisoformat(w["best"]).strftime("%d %b %H:%M") for w in slips[:3]) + " UTC.")
    alt = [r for r in region_best if r["site_id"] != reg.get("id")]
    if alt:
        expl.append(f"Backup region: {alt[0]['name']} (best {alt[0]['success_probability']*100:.0f}%, {alt[0]['viable_share']*100:.0f}% of its area viable).")
    return {
        "available": True,
        "matrix": {"site_ids": [r.get("id") for r in regions], "times": [t.isoformat() for t in times], "success": np.round(score, 4).tolist(), "best_success": np.round(best, 4).tolist(), "viable_share": np.round(viable, 4).tolist()},
        "primary": {
            "site_id": reg.get("id"), "name": reg.get("name"), "lat": reg["lat"], "lon": reg["lon"], "half_size_m": reg.get("half_size_m", 7500.0),
            "landing_time": times[j].isoformat(), "success_probability": round(float(best[i, j]), 4), "viable_share": round(float(viable[i, j]), 4),
            "zone": zones[0] if zones else None,
        },  # fmt: skip
        "zones": zones,
        "region_stats": st,
        "best_times": region_windows,
        "site_best": region_best,
        "alternatives": alt[:3],
        "explanation": expl,
        "no_viable": bool(not zones or best[i, j] < ZONE_SUCCESS),
        "model": {"version": ai._load()["ck"]["version"]},
    }
