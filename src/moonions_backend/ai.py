"""AI inference (v2): landing-success = AI touchdown safety × deterministic light/power/Earth link.

* Touchdown: ensemble of physics-informed residual networks (ml/train_td.py) → mean + uncertainty.
* Light at landing, power through the stay, Earth link: exact physics from JPL ephemeris, terrain
  horizons and the battery model (ml/simulate.time_outcomes) — never learned.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

from .backups import compass, divert_cost
from .config import MODELS
from .dem import destination, distance_bearing, offset_latlon
from .ephemeris import to_utc
from .horizon import Horizon

log = logging.getLogger(__name__)
CKPT = MODELS / "td_ensemble.pt"
DOMAIN_MAX_LAT = -80.6
HEADS = ["touchdown", "lit0", "energy_ok", "comms_ok"]
FRIENDLY = {
    "slope_deg": ("Average tilt of landing zone", "°"),
    "slope_p90_deg": ("Steepest ground in zone", "°"),
    "roughness_m": ("Surface roughness", "m"),
    "unsafe_fraction": ("Share of steep ground (80 m map)", ""),
    "relief_m": ("Height variation in zone", "m"),
    "phys80_td": ("Physics estimate on the 80 m map", ""),
    "phys80_direct": ("Chance no divert is needed (80 m)", ""),
    "phys80_tip": ("Tip-over risk (80 m map)", ""),
    "crater_density": ("Crater density nearby", "/1000 km²"),
    "rim_km": ("Distance to nearest crater rim", "km"),
    "inside_crater": ("Inside a mapped crater", ""),
    "deflection_deg": ("Gravity tilt (GRAIL)", "°"),
    "colat": ("Distance from the pole", "°"),
    "terrain": ("Terrain shape (AI vision)", ""),
}


# ------------------------------------------------------------------ model


@lru_cache(maxsize=1)
def _load():
    if not CKPT.exists():
        return None
    try:
        import torch

        from .ml.td_model import TouchdownNet

        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        variants = ck.get("all_variants") or {ck["variant"]: {"members": ck["members"], "use_cnn": ck["use_cnn"], "emb_dim": ck["emb_dim"]}}
        models = {}
        for v, info in variants.items():
            ms = []
            for sd in info["members"]:
                m = TouchdownNet(n_tab=len(ck["tab_features"]), in_ch=ck["in_ch"], use_cnn=info["use_cnn"], emb_dim=info["emb_dim"])
                m.load_state_dict(sd)
                ms.append(m.eval().to(dev))
            models[v] = (ms, info)
        return {"ck": ck, "models": models, "dev": dev}
    except Exception as exc:  # noqa: BLE001
        log.warning("touchdown model unavailable: %s", exc)
        return None


def available() -> bool:
    return _load() is not None


def model_card() -> dict:
    out: dict = {"available": available()}
    card = MODELS / "success_model_card.json"
    if card.exists() and available():
        out["success_model"] = json.loads(card.read_text())
    return out


def _fast_variant() -> str:
    L = _load()
    return "cnn" if L and "cnn" in L["models"] else (L["ck"]["variant"] if L else "cnn")


# ------------------------------------------------------------------ features


@lru_cache(maxsize=50000)
def _site(lat_q: float, lon_q: float):
    from .ml.td_dataset import TAB, patch80, planning_features

    pf = planning_features(lat_q, lon_q)
    return (
        np.array([pf[k] for k in TAB], np.float32),
        patch80(lat_q, lon_q).astype(np.float32),
        Horizon(lat_q, lon_q).profile.astype(np.float32),
        float(pf["elevation_m"]),
    )


@lru_cache(maxsize=4096)
def _emb_cached(lat_q: float, lon_q: float) -> np.ndarray:
    from .ml.lfm_embed import embed

    e = embed([lat_q], [lon_q])
    return np.concatenate([e["center"], e["global"]], 1)[0].astype(np.float32)


def _sites(lats, lons):
    rows = [_site(round(float(a), 6), round(float(b), 6)) for a, b in zip(lats, lons)]
    return {
        "tab": np.stack([r[0] for r in rows]),
        "patch": np.stack([r[1] for r in rows]),
        "horizon": np.stack([r[2] for r in rows]),
        "elev": np.array([r[3] for r in rows], np.float32),
    }


def _horizon_at(profiles: np.ndarray, az: np.ndarray) -> np.ndarray:
    n = profiles.shape[1]  # azimuth bins (2° for bulk grids, 0.5° for detailed site analysis)
    step = 360.0 / n
    x = np.mod(az, 360.0) / step
    i0 = np.floor(x).astype(int) % n
    f = x - np.floor(x)
    rows = np.arange(profiles.shape[0])[:, None]
    return profiles[rows, i0] * (1 - f) + profiles[rows, (i0 + 1) % n] * f


def physics_gates(lats, lons, elev, horizon, times: list[datetime]) -> dict[str, np.ndarray]:
    """Light / power / Earth-link outcomes for N sites x M touchdown times → arrays (N, M)."""
    from .ml.simulate import DEFAULT, time_outcomes
    from .ml.sky import body_vectors_km, sky_many

    steps = int(DEFAULT.mission_hours / DEFAULT.step_hours) + 1
    stamps = [to_utc(t) + timedelta(hours=DEFAULT.step_hours * k) for t in times for k in range(steps)]
    allv = body_vectors_km(stamps)
    keys = ("lit0", "energy_ok", "comms_ok", "lit_frac", "earth_frac", "min_soc", "longest_dark_h", "sun_clear0", "earth_clear0")
    out = {k: np.zeros((len(lats), len(times)), np.float32) for k in keys}
    for j in range(len(times)):
        v = {k: allv[k][j * steps : (j + 1) * steps] for k in allv}
        s = sky_many(np.asarray(lats), np.asarray(lons), v, height_m=elev)
        sc = s["sun_el"] - _horizon_at(horizon, s["sun_az"])
        ec = s["earth_el"] - _horizon_at(horizon, s["earth_az"])
        from .conditions import EARTH_RADIUS_KM, SUN_RADIUS_KM, angular_radius, covered_fraction, disk_fraction_above

        sun_r = angular_radius(SUN_RADIUS_KM, s["sun_dist_km"])
        frac = disk_fraction_above(sc, sun_r) * (1 - covered_fraction(s["sun_earth_sep_deg"], sun_r, angular_radius(EARTH_RADIUS_KM, s["earth_dist_km"])))
        o = time_outcomes(sc, s["sun_el"], ec, DEFAULT, sun_frac=frac)
        o["sun_clear0"], o["earth_clear0"] = sc[:, 0], ec[:, 0]
        for k in keys:
            out[k][:, j] = o[k]
    return out


def predict_td(tab: np.ndarray, patch: np.ndarray, lats=None, lons=None, variant: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble touchdown probability (mean, std) for N sites."""
    import torch

    from .ml.td_dataset import TAB

    L = _load()
    ck = L["ck"]
    variant = variant or ck["variant"]
    members, info = L["models"][variant]
    dev = L["dev"]
    x = (tab - ck["mu"]) / ck["sd"]
    emb = _embs(lats, lons) if info["emb_dim"] else None
    pr_td = tab[:, TAB.index("phys80_td")]
    pr_dir = tab[:, TAB.index("phys80_direct")]
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)  # noqa: E731
    preds = []
    with torch.no_grad():
        for m in members:
            out = []
            for i in range(0, len(x), 4096):
                s = slice(i, i + 4096)
                o = m(T(patch[s]) if info["use_cnn"] else None, T(x[s]), T(emb[s]) if emb is not None else None, T(pr_td[s]), T(pr_dir[s]))
                out.append(torch.sigmoid(o[:, 0]).cpu().numpy())
            preds.append(np.concatenate(out))
    P = np.stack(preds)
    return P.mean(0), P.std(0)


def _embs(lats, lons):
    return np.stack([_emb_cached(round(float(a), 6), round(float(b), 6)) for a, b in zip(lats, lons)])


def in_domain(lat: float) -> tuple[bool, str | None]:
    if lat <= DOMAIN_MAX_LAT:
        return True, None
    return False, "Outside the south-polar region the AI was trained on; touchdown numbers are indicative."


# ------------------------------------------------------------------ endpoints


def assess(lat: float, lon: float, when: datetime) -> dict:
    L = _load()
    if L is None:
        return {"available": False}
    from .ml.td_dataset import TAB

    ck = L["ck"]
    f = _sites([lat], [lon])
    td, sd = predict_td(f["tab"], f["patch"], [lat], [lon])
    g = physics_gates([lat], [lon], f["elev"], f["horizon"], [when])
    heads = {"touchdown": float(td[0]), "lit0": float(g["lit0"][0, 0]), "energy_ok": float(g["energy_ok"][0, 0]), "comms_ok": float(g["comms_ok"][0, 0])}
    succ = float(np.prod(list(heads.values())))

    # Touchdown explanation by occlusion: reset one input to its training average and see how the
    # predicted touchdown probability moves. impact > 0 means that input helps.
    reps = np.repeat(f["tab"], len(TAB), 0)
    for i in range(len(TAB)):
        if TAB[i] in ("phys80_td", "phys80_direct"):
            continue  # the physics prior is the model's anchor; reported separately
        reps[i, i] = ck["mu"][i]
    occ, _ = predict_td(reps, np.repeat(f["patch"], len(TAB), 0), [lat] * len(TAB), [lon] * len(TAB))
    drivers = []
    for i, name in enumerate(TAB):
        if name in ("phys80_td", "phys80_direct"):
            continue
        label, unit = FRIENDLY.get(name, (name, ""))
        drivers.append({"feature": name, "label": label, "value": round(float(f["tab"][0, i]), 3), "unit": unit, "impact": round(float(td[0] - occ[i]), 4)})
    blank, _ = predict_td(f["tab"], np.zeros_like(f["patch"]), [lat], [lon])
    drivers.append({"feature": "terrain", "label": FRIENDLY["terrain"][0], "value": None, "unit": "", "impact": round(float(td[0] - blank[0]), 4)})
    drivers.sort(key=lambda d: -abs(d["impact"]))

    checks = [
        {"key": "lit0", "ok": bool(heads["lit0"]), "label": "Light at landing", "detail": f"Sun {g['sun_clear0'][0,0]:+.1f}° vs skyline"},
        {"key": "energy_ok", "ok": bool(heads["energy_ok"]), "label": "Power through the stay", "detail": f"lit {g['lit_frac'][0,0]*100:.0f}% of the stay, lowest battery {g['min_soc'][0,0]*100:.0f}%"},
        {"key": "comms_ok", "ok": bool(heads["comms_ok"]), "label": "Earth link", "detail": f"Earth in view {g['earth_frac'][0,0]*100:.0f}% of the stay"},
    ]
    dom, note = in_domain(lat)
    from .scoring import EvalParams, evaluate_site

    rule = evaluate_site(lat, lon, EvalParams(landing_time=when, profile="safety-first"), with_backups=False)["overall"]
    return {
        "available": True,
        "in_domain": dom,
        "domain_note": note,
        "success_probability": round(succ, 4),
        "heads": {k: round(v, 4) for k, v in heads.items()},
        "touchdown_uncertainty": round(float(sd[0]), 4),
        "physics_prior": round(float(f["tab"][0, TAB.index("phys80_td")]), 4),
        "checks": checks,
        "rule_score": rule,
        "drivers": drivers[:7],
        "model": {"version": ck["version"], "variant": ck["variant"], "foundation_model": "NASA-IBM LFM" if ck["emb_dim"] else None},
    }


def site_map(lat: float, lon: float, when: datetime, half_size_m: float = 3000.0, cells: int = 25) -> dict:
    if _load() is None:
        return {"available": False}
    axis = np.linspace(-half_size_m, half_size_m, cells)
    E, N = np.meshgrid(axis, axis[::-1])
    plat, plon = offset_latlon(lat, lon, E.ravel(), N.ravel())
    f = _sites(plat, plon)
    td, sd = predict_td(f["tab"], f["patch"], plat, plon, variant=_fast_variant())
    g = physics_gates(plat, plon, f["elev"], f["horizon"], [when])
    gate = (g["lit0"] * g["energy_ok"] * g["comms_ok"])[:, 0]
    return {
        "available": True,
        "half_size_m": half_size_m,
        "cells": cells,
        "success": np.round(td * gate, 3).tolist(),
        "touchdown": np.round(td, 3).tolist(),
        "uncertainty": np.round(sd, 3).tolist(),
        "model": _load()["ck"]["version"] + f" ({_fast_variant()})",
    }


def windows(times: list[datetime], succ: np.ndarray, step_hours: float, threshold: float = 0.5, top: int = 5) -> list[dict]:
    out, i, n = [], 0, len(succ)
    while i < n:
        if succ[i] >= threshold:
            j = i
            while j + 1 < n and succ[j + 1] >= threshold:
                j += 1
            k = i + int(np.argmax(succ[i : j + 1]))
            out.append(
                {
                    "start": times[i].isoformat(),
                    "end": (times[j] + timedelta(hours=step_hours)).isoformat(),
                    "best": times[k].isoformat(),
                    "best_success": round(float(succ[k]), 4),
                    "hours_open": round(float((j - i + 1) * step_hours), 1),
                }
            )
            i = j + 1
        else:
            i += 1
    out.sort(key=lambda w: -w["best_success"])
    return out[:top]


def timeline(lat: float, lon: float, start: datetime, days: float = 30.0, step_hours: float = 6.0) -> dict:
    if _load() is None:
        return {"available": False}
    times = [to_utc(start) + timedelta(hours=step_hours * k) for k in range(int(days * 24 / step_hours) + 1)]
    f = _sites([lat], [lon])
    td, _ = predict_td(f["tab"], f["patch"], [lat], [lon])
    g = physics_gates([lat], [lon], f["elev"], f["horizon"], times)
    succ = td[0] * g["lit0"][0] * g["energy_ok"][0] * g["comms_ok"][0]
    return {
        "available": True,
        "times": [t.isoformat() for t in times],
        "success": np.round(succ, 4).tolist(),
        "heads": {"touchdown": [round(float(td[0]), 4)] * len(times), "lit0": g["lit0"][0].tolist(), "energy_ok": g["energy_ok"][0].tolist(), "comms_ok": g["comms_ok"][0].tolist()},
        "windows": windows(times, succ, step_hours),
        "step_hours": step_hours,
    }


def recommend(sites: list[dict], start: datetime, days: float, step_hours: float, divert_radius_m: float = 3000.0, n_backups: int = 4) -> dict:
    L = _load()
    if L is None:
        return {"available": False}
    times = [to_utc(start) + timedelta(hours=step_hours * k) for k in range(int(days * 24 / step_hours) + 1)]
    lats = np.array([s["lat"] for s in sites])
    lons = np.array([s["lon"] for s in sites])
    f = _sites(lats, lons)
    td, td_sd = predict_td(f["tab"], f["patch"], lats, lons)
    g = physics_gates(lats, lons, f["elev"], f["horizon"], times)
    gate = g["lit0"] * g["energy_ok"] * g["comms_ok"]
    succ = td[:, None] * gate
    N, M = succ.shape
    i, j = np.unravel_index(np.argmax(succ + 1e-6 * td[:, None]), succ.shape)
    best = float(succ[i, j])
    prim = sites[i]
    ph = {"touchdown": round(float(td[i]), 4), "lit0": float(g["lit0"][i, j]), "energy_ok": float(g["energy_ok"][i, j]), "comms_ok": float(g["comms_ok"][i, j])}

    # backup sites: AI-scored ring candidates around the primary at the chosen time, diverse directions
    rings = [r for r in (250, 400, 600, 850, 1100, 1500, 2000, 2500, 3000, 4000, 5000) if r <= divert_radius_m]
    pts = [(r, b) for r in rings for b in np.arange(0, 360, 30)]
    blat, blon = destination(prim["lat"], prim["lon"], [b for _, b in pts], [r for r, _ in pts])
    fb = _sites(blat, blon)
    tdb, _ = predict_td(fb["tab"], fb["patch"], blat, blon, variant=_fast_variant())
    gb = physics_gates(blat, blon, fb["elev"], fb["horizon"], [times[j]])
    gateb = (gb["lit0"] * gb["energy_ok"] * gb["comms_ok"])[:, 0]
    sb = tdb * gateb
    order = np.argsort(-(sb - 0.03 * np.array([r for r, _ in pts]) / max(divert_radius_m, 1)))
    backups, used = [], set()
    for k in order:
        r, b = pts[k]
        sector = int(((b % 360) + 22.5) // 45) % 8
        if sector in used or sb[k] < 0.3 or any(distance_bearing(blat[k], blon[k], x["lat"], x["lon"])[0] < 400 for x in backups):
            continue
        used.add(sector)
        why = []
        if tdb[k] > td[i] + 0.02:
            why.append("safer ground than the primary")
        elif tdb[k] > 0.9:
            why.append("safe ground")
        why.append("same sunlight and Earth link" if gateb[k] else "")
        backups.append(
            {
                "rank": len(backups) + 1,
                "lat": float(blat[k]),
                "lon": float(blon[k]),
                "distance_m": float(r),
                "bearing_deg": float(b),
                "direction": compass(b),
                "success_probability": round(float(sb[k]), 4),
                "heads": {"touchdown": round(float(tdb[k]), 4), "lit0": float(gb["lit0"][k, 0]), "energy_ok": float(gb["energy_ok"][k, 0]), "comms_ok": float(gb["comms_ok"][k, 0])},
                "decide_by_s": divert_cost(r)["decide_by_s"],
                "reason": ", ".join(w for w in why if w).capitalize() or "Nearest acceptable alternative",
            }
        )
        if len(backups) >= n_backups:
            break

    best_times = windows(times, succ[i], step_hours)
    for w in best_times:
        w["is_primary"] = w["best"] == times[j].isoformat()
    site_best = sorted(
        (
            {"site_id": sites[a].get("id"), "name": sites[a].get("name"), "landing_time": times[int(np.argmax(succ[a]))].isoformat(), "success_probability": round(float(succ[a].max()), 4), "viable_slots": int((succ[a] >= 0.5).sum())}
            for a in range(N)
        ),
        key=lambda x: -x["success_probability"],
    )
    alts = [{"site_id": x["site_id"], "name": x["name"], "landing_time": x["landing_time"], "success_probability": x["success_probability"]} for x in site_best if x["site_id"] != prim.get("id")][:3]

    when_str = times[j].strftime("%d %b %H:%M UTC")
    expl = [f"{prim.get('name', 'Site')} on {when_str}: highest predicted landing success ({best*100:.0f}%) of {N} sites × {M} touchdown times."]
    expl.append(f"AI touchdown safety {td[i]*100:.0f}% (±{td_sd[i]*100:.0f}); light, power and Earth link are confirmed by physics for this time.")
    good = int((succ[i] >= 0.5).sum())
    expl.append(f"This site has {good} viable touchdown slots in the period.")
    slips = [w for w in best_times if not w["is_primary"]]
    if slips:
        expl.append(f"If the date slips, the next best windows here open {', '.join(datetime.fromisoformat(w['best']).strftime('%d %b %H:%M') for w in slips[:3])} UTC.")
    if alts:
        expl.append(f"Best alternative site: {alts[0]['name']} at {alts[0]['success_probability']*100:.0f}%.")
    if backups:
        expl.append(f"{len(backups)} AI-checked backup sites within {divert_radius_m/1000:g} km, each in a different direction.")
    dom, note = in_domain(prim["lat"])
    if not dom:
        expl.append(note)
    return {
        "available": True,
        "matrix": {"site_ids": [s.get("id") for s in sites], "times": [t.isoformat() for t in times], "success": np.round(succ, 4).tolist()},
        "primary": {"site_id": prim.get("id"), "name": prim.get("name"), "lat": prim["lat"], "lon": prim["lon"], "landing_time": times[j].isoformat(), "success_probability": round(best, 4), "heads": ph},
        "backups": backups,
        "best_times": best_times,
        "site_best": site_best,
        "alternatives": alts,
        "explanation": expl,
        "no_viable": best < 0.5,
        "model": {"version": L["ck"]["version"], "variant": L["ck"]["variant"]},
    }
