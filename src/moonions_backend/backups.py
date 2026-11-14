"""Backup landing sites close to a primary target.

If the lander can't set down at the primary point late in descent (hazard detected, dust,
navigation drift), it needs pre-vetted alternatives within divert reach. This module finds
them, makes sure they don't share the same weakness (spread in distance and direction), and
estimates what diverting to each costs.
"""

from __future__ import annotations

import numpy as np

from .config import MOON_G
from .craters import get_craters
from .dem import destination, distance_bearing, get_dem
from .terrain import analyze_zone

TIERS = [
    ("near", 500.0, "Last-second hop"),
    ("mid", 1500.0, "Short divert"),
    ("far", float("inf"), "Early retarget"),
]
LATERAL_ACCEL = 1.5  # m/s^2 sustained sideways acceleration while hovering
HOVER_RESERVE_S = 75.0  # fuel reserve set aside for hazard avoidance (seconds of hover)
RETARGET_DV_BUDGET = 60.0  # m/s of propellant set aside for retargeting during approach


def divert_cost(dist_m: float) -> dict:
    """Cost of switching to a backup site.

    * Retarget during approach (still descending): a bang-bang sideways shift over the
      remaining time T needs Δv = 4·d/T, so with budget B the decision must be made at least
      T = 4·d/B seconds before touchdown. Gravity losses are already part of the descent.
    * Hover hop (already hovering over the primary): accelerate half-way, brake half-way:
      t = 2·√(d/a), Δv = 2·√(d·a) + g·t.
    """
    decide_by = 4 * dist_m / RETARGET_DV_BUDGET
    t = 2 * np.sqrt(dist_m / LATERAL_ACCEL)
    dv = 2 * np.sqrt(dist_m * LATERAL_ACCEL) + MOON_G * t
    return {
        "decide_by_s": round(float(decide_by), 0),
        "hover_seconds": round(float(t), 1),
        "delta_v_mps": round(float(dv), 1),
        "within_reserve": bool(t <= HOVER_RESERVE_S),
        "reserve_used_pct": round(float(100 * t / HOVER_RESERVE_S), 0),
    }


def tier_for(dist_m: float) -> tuple[str, str]:
    for key, lim, label in TIERS:
        if dist_m <= lim:
            return key, label
    return TIERS[-1][0], TIERS[-1][2]


def _ring_samples(lat: float, lon: float, divert_m: float, min_step: float):
    radii = [r for r in (250, 400, 600, 850, 1100, 1500, 2000, 2500, 3000, 4000, 5000) if r <= divert_m]
    radii = [r for r in radii if r >= min_step] or [min_step]
    pts = []
    for r in radii:
        n_b = int(np.clip(2 * np.pi * r / max(min_step, 250), 8, 36))
        for b in np.linspace(0, 360, n_b, endpoint=False):
            pts.append((r, b))
    pts = np.array(pts)
    plat, plon = destination(lat, lon, pts[:, 1], pts[:, 0])
    return plat, plon, pts[:, 0], pts[:, 1]


def _quick_score(t, crater_info) -> float:
    from .scoring import smooth

    s = 0.5 * (1 - smooth(t.slope_p90_deg, 3, 15)) + 0.3 * (1 - smooth(t.unsafe_fraction, 0, 0.6)) + 0.2 * (1 - smooth(t.roughness_m, 1, 25))
    if crater_info is not None:
        rim = crater_info["nearest_rim_km"]
        s *= 0.75 + 0.25 * (1.0 if rim is None else float(smooth(rim, 0.1, 1.0)))
    if t.slope_deg > 15:
        s *= 0.2
    return float(100 * s)


def plan_backups(lat: float, lon: float, p) -> dict:
    from .scoring import Factor, evaluate_site

    dem = get_dem()
    src = dem.source_for(lat)
    res = src.resolution_m
    # Backups must be separated by at least a DEM cell and a landing zone so they're distinct
    min_sep = max(2 * p.zone_radius_m, 1.2 * res, 300.0)
    zone = max(p.zone_radius_m * 0.6, 150.0)
    plat, plon, prad, pbear = _ring_samples(lat, lon, p.divert_radius_m, min_step=max(250.0, 0.8 * res))
    db = get_craters()

    cands = []
    for la, lo, r, b in zip(plat, plon, prad, pbear):
        t = analyze_zone(float(la), float(lo), zone, source=src)
        ci = db.around(float(la), float(lo), 1500) if db else None
        q = _quick_score(t, ci)
        # prefer closer sites: cheaper and keeps the backup inside the surveyed area
        cost_pen = 8.0 * (r / max(p.divert_radius_m, 1.0))
        cands.append({"lat": float(la), "lon": float(lon_wrap(lo)), "dist": float(r), "bearing": float(b), "terrain": t, "q": q, "rank_key": q - cost_pen})

    cands.sort(key=lambda c: -c["rank_key"])
    chosen: list[dict] = []
    sectors_used: dict[int, int] = {}
    for c in cands:
        if len(chosen) >= p.n_backups * 2:  # pre-select twice as many, full-evaluate, then trim
            break
        if c["q"] < 35:
            continue
        sep_ok = all(distance_bearing(c["lat"], c["lon"], o["lat"], o["lon"])[0] >= min_sep for o in chosen)
        if not sep_ok:
            continue
        # one backup per compass direction (8 sectors) so a single local problem (dust plume,
        # boulder field, shadow) can't knock out every option at once
        sector = int(((c["bearing"] % 360) + 22.5) // 45) % 8
        if sectors_used.get(sector, 0) >= (1 if len(sectors_used) < 8 else 2):
            continue
        sectors_used[sector] = sectors_used.get(sector, 0) + 1
        chosen.append(c)

    sites = []
    for c in chosen:
        ev = evaluate_site(c["lat"], c["lon"], p, with_backups=False, terrain=c["terrain"])
        tier, tier_label = tier_for(c["dist"])
        cost = divert_cost(c["dist"])
        sites.append(
            {
                "lat": c["lat"],
                "lon": c["lon"],
                "distance_m": round(c["dist"], 0),
                "bearing_deg": round(c["bearing"], 0),
                "direction": compass(c["bearing"]),
                "tier": tier,
                "tier_label": tier_label,
                "divert": cost,
                "overall": ev["overall"],
                "verdict": ev["verdict"],
                "factors": ev["factors"],
                "terrain": ev["terrain"],
                "rank_score": ev["overall"] - 6 * (0 if cost["within_reserve"] else 1) - 6 * c["dist"] / max(p.divert_radius_m, 1),
            }
        )
    sites.sort(key=lambda s: -s["rank_score"])
    sites = [s for s in sites if s["verdict"] != "no-go"][: p.n_backups] or sites[: min(2, len(sites))]
    for i, s in enumerate(sites, 1):
        s["rank"] = i
        s["id"] = f"B{i}"

    usable = [s for s in sites if s["verdict"] != "no-go"]
    best = max((s["overall"] for s in usable), default=0.0)
    count_q = min(1.0, len(usable) / 3)
    hop_q = 1.0 if any(s["divert"]["within_reserve"] for s in usable) else 0.8  # can we still hop at the last second?
    spread = len({int(s["bearing_deg"] // 90) for s in usable})
    spread_q = min(1.0, spread / 3)
    score = 100 * (0.4 * count_q + 0.4 * (best / 100) * hop_q + 0.2 * spread_q)
    if not usable:
        msg = "No safe fallback site within divert reach: an aborted touchdown means no landing."
    else:
        b0 = usable[0]
        msg = f"{len(usable)} safe fallback sites in reach; best is {b0['distance_m']:.0f} m {b0['direction']} (decide by T-{b0['divert']['decide_by_s']:.0f} s)."
    factor = Factor(
        "backups",
        score,
        msg,
        {"usable": len(usable), "candidates_screened": len(cands), "min_separation_m": round(min_sep), "dem_resolution_m": round(res, 1)},
        hard_fail=False,
    )
    return {
        "sites": sites,
        "factor": factor,
        "divert": {
            "radius_m": p.divert_radius_m,
            "lateral_accel_mps2": LATERAL_ACCEL,
            "hover_reserve_s": HOVER_RESERVE_S,
            "retarget_dv_budget_mps": RETARGET_DV_BUDGET,
            "tiers": [{"key": k, "max_m": (None if lim == float("inf") else lim), "label": lab} for k, lim, lab in TIERS],
        },
    }


def lon_wrap(lon):
    return (float(lon) + 180.0) % 360.0 - 180.0


def compass(b: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((b % 360) + 22.5) // 45) % 8]


def suggest_in_region(lat: float, lon: float, region_radius_m: float, p, n: int = 4) -> list[dict]:
    """Search a region (e.g. a 15x15 km Artemis region) for the best primary targets.

    Returns several well-separated primaries so that the mission has independent options at
    the region level too, each evaluated with its own nearby backups.
    """
    from .scoring import evaluate_site

    dem = get_dem()
    src = dem.source_for(lat)
    step = max(src.resolution_m * 2, region_radius_m / 12)
    axis = np.arange(-region_radius_m, region_radius_m + 1, step)
    E, N = np.meshgrid(axis, axis)
    mask = np.hypot(E, N) <= region_radius_m
    from .dem import offset_latlon

    plat, plon = offset_latlon(lat, lon, E[mask], N[mask])
    db = get_craters()
    scored = []
    for la, lo in zip(plat, plon):
        t = analyze_zone(float(la), float(lo), p.zone_radius_m, source=src)
        ci = db.around(float(la), float(lo), 1500) if db else None
        scored.append((_quick_score(t, ci), float(la), float(lo), t))
    scored.sort(key=lambda x: -x[0])
    picks: list[tuple] = []
    min_sep = max(p.divert_radius_m * 0.8, 1500.0)
    for s in scored:
        if all(distance_bearing(s[1], s[2], q[1], q[2])[0] >= min_sep for q in picks):
            picks.append(s)
        if len(picks) >= n * 2:
            break
    # full evaluation of the shortlisted primaries, but cheap backups planning on the top n
    evals = [evaluate_site(la, lo, p, with_backups=False, terrain=t) for _, la, lo, t in picks]
    order = np.argsort([-e["overall"] for e in evals])[:n]
    return [evaluate_site(evals[i]["lat"], evals[i]["lon"], p, with_backups=True, terrain=picks[i][3]) for i in order]
