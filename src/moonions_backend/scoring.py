"""Landing suitability scoring.

Turns terrain, illumination, Earth visibility and crater data into plain-language factor
scores (0-100), an overall score and a verdict (Go / Caution / No-go).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .craters import get_craters
from .ephemeris import sky_from_site, time_range, to_utc
from .horizon import Horizon
from .terrain import MAX_SLOPE_DEG, SAFE_SLOPE_DEG, TerrainStats, analyze_zone

PROFILES: dict[str, dict[str, float]] = {
    "balanced": {"ground": 0.28, "sunlight": 0.2, "earth": 0.14, "craters": 0.12, "backups": 0.16, "science": 0.10},
    "safety-first": {"ground": 0.34, "sunlight": 0.18, "earth": 0.14, "craters": 0.14, "backups": 0.18, "science": 0.02},
    "science-first": {"ground": 0.22, "sunlight": 0.16, "earth": 0.1, "craters": 0.08, "backups": 0.14, "science": 0.30},
}

FACTOR_LABELS = {
    "ground": "Ground safety",
    "sunlight": "Sunlight",
    "earth": "Earth contact",
    "craters": "Crater risk",
    "backups": "Backup options",
    "science": "Science value",
}

IDEAL_SUN = (7.0, 25.0)  # good shadows for camera-based hazard detection


def sun_angle_quality(el):
    """How good the Sun angle is for camera-based hazard detection (0..1).

    Below ~3° shadows get extremely long (typical at the poles, where it never gets much
    higher), 7-25° is ideal, high Sun washes out shadows.
    """
    el = np.asarray(el, dtype=float)
    rising = 0.6 * smooth(el, 0.0, 3.0) + 0.4 * smooth(el, 3.0, IDEAL_SUN[0])
    return rising * (1 - 0.6 * smooth(el, IDEAL_SUN[1], 60.0))


def smooth(x, lo, hi):
    """0 at lo, 1 at hi, smoothstep in between (works for lo > hi too)."""
    t = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


@dataclass
class Factor:
    key: str
    score: float | None
    summary: str
    details: dict = field(default_factory=dict)
    hard_fail: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": FACTOR_LABELS[self.key],
            "score": None if self.score is None else round(float(self.score), 1),
            "summary": self.summary,
            "details": self.details,
            "hard_fail": self.hard_fail,
        }


# ---------------------------------------------------------------- factors


def ground_factor(t: TerrainStats, ai: dict | None) -> Factor:
    s_slope = 1 - smooth(t.slope_p90_deg, 3.0, MAX_SLOPE_DEG)
    s_unsafe = 1 - smooth(t.unsafe_fraction, 0.0, 0.6)
    s_rough = 1 - smooth(t.roughness_m, 1.0, 25.0)
    score = 100 * (0.5 * s_slope + 0.3 * s_unsafe + 0.2 * s_rough)
    if ai is not None:
        score = 0.7 * score + 0.3 * 100 * ai["p_safe"]
    hard = t.slope_deg > MAX_SLOPE_DEG
    if hard:
        msg = f"Too steep: average tilt {t.slope_deg:.0f}°, the lander could tip over."
    elif t.slope_p90_deg < SAFE_SLOPE_DEG and t.unsafe_fraction < 0.1:
        msg = f"Flat and even ground (tilt ≈{t.slope_deg:.1f}°)."
    elif t.unsafe_fraction < 0.35:
        msg = f"Mostly safe ground; {t.unsafe_fraction*100:.0f}% of the zone is too steep."
    else:
        msg = f"Uneven ground: {t.unsafe_fraction*100:.0f}% of the zone is too steep to land."
    d = t.to_dict()
    if ai is not None:
        d["ai"] = ai
    return Factor("ground", score, msg, d, hard)


def sunlight_factor(sky: dict, hz: Horizon, mission_steps: int) -> Factor:
    sun_clear = sky["sun_clear"] if "sun_clear" in sky else sky["sun_el"] - hz.at(sky["sun_az"])
    lit = sky["lit"] if "lit" in sky else sun_clear > 0.25  # most of the solar disk visible
    el0 = float(sky["sun_el"][0])
    clear0 = float(sun_clear[0])
    dark0 = not bool(lit[0])
    lit_frac = float(np.mean(lit[:mission_steps])) if mission_steps else float(lit[0])
    if dark0:
        angle_q = 0.0
    else:
        angle_q = float(sun_angle_quality(el0))
    score = 100 * (0.55 * angle_q + 0.45 * lit_frac)
    hard = dark0
    if hard:
        msg = "Dark at touchdown: the Sun is behind the horizon or mountains."
    elif el0 < 3.0:
        msg = f"Sun just {clear0:.1f}° above the skyline: very long shadows. Lit {lit_frac*100:.0f}% of the stay."
    elif el0 < IDEAL_SUN[0]:
        msg = f"Low Sun ({el0:.1f}°): long shadows can hide hazards. Lit {lit_frac*100:.0f}% of the stay."
    elif el0 > IDEAL_SUN[1]:
        msg = f"High Sun ({el0:.0f}°): weak shadows make rocks harder to spot. Lit {lit_frac*100:.0f}% of the stay."
    else:
        msg = f"Good lighting ({el0:.0f}° Sun). Lit {lit_frac*100:.0f}% of the stay."
    return Factor(
        "sunlight",
        score,
        msg,
        {
            "sun_elevation_deg": round(el0, 2),
            "sun_azimuth_deg": round(float(sky["sun_az"][0]), 1),
            "sun_clearance_deg": round(clear0, 2),
            "terrain_horizon_deg": round(float(hz.at(sky["sun_az"][0])), 2),
            "lit_fraction": round(lit_frac, 3),
        },
        hard,
    )


def earth_factor(sky: dict, hz: Horizon, mission_steps: int) -> Factor:
    clear = sky["earth_clear"] if "earth_clear" in sky else sky["earth_el"] - hz.at(sky["earth_az"])
    vis = sky["earth_ok"] if "earth_ok" in sky else clear > 0.5
    vis0 = bool(vis[0])
    frac = float(np.mean(vis[:mission_steps])) if mission_steps else float(vis0)
    margin = float(clear[0])
    score = 100 * (0.45 * float(vis0) * float(smooth(margin, 0.5, 8.0) * 0.4 + 0.6) + 0.55 * frac)
    if not vis0:
        msg = "No direct line to Earth at touchdown: needs a relay satellite."
    elif frac > 0.95:
        msg = f"Earth in view for the whole stay ({margin:.1f}° above the skyline)."
    else:
        msg = f"Earth in view at touchdown, {frac*100:.0f}% of the stay."
    return Factor(
        "earth",
        score,
        msg,
        {
            "earth_elevation_deg": round(float(sky["earth_el"][0]), 2),
            "earth_azimuth_deg": round(float(sky["earth_az"][0]), 1),
            "earth_clearance_deg": round(margin, 2),
            "terrain_horizon_deg": round(float(hz.at(sky["earth_az"][0])), 2),
            "visible_fraction": round(frac, 3),
        },
    )


def crater_factor(lat: float, lon: float) -> Factor:
    db = get_craters()
    if db is None:
        return Factor("craters", None, "Crater database unavailable; crater risk not assessed.", {"available": False})
    c5 = db.around(lat, lon, 5_000)
    c1 = db.around(lat, lon, 1_500)
    dens = c5["density_per_1000km2"]
    s_dens = 1 - smooth(dens, 25, 250)  # global mean ≈ 34 craters ≥1 km per 1000 km²
    rim = c1["nearest_rim_km"]
    s_rim = 1.0 if rim is None else float(smooth(rim, 0.1, 1.2))
    score = 100 * (0.6 * s_dens + 0.4 * s_rim)
    if rim is not None and rim < 0.3:
        msg = "Right on a crater rim: steep walls and boulders likely."
    elif dens > 150:
        msg = f"Heavily cratered area ({c5['count']} mapped craters within 5 km)."
    elif dens > 100:
        msg = f"Moderately cratered ({c5['count']} mapped craters within 5 km)."
    else:
        msg = f"Few mapped craters nearby ({c5['count']} within 5 km)."
    return Factor("craters", score, msg, {"within_5km": c5, "within_1_5km": c1})


def science_factor(lat: float, tags: list[str] | None = None) -> Factor:
    return Factor(
        "science", None,
        "Science value not assessed: no cold-trap, ice-abundance or science-target layers are integrated.",
        {"available": False},
    )


# ---------------------------------------------------------------- evaluation


@dataclass
class EvalParams:
    landing_time: datetime
    mission_hours: float = 156.0  # ~6.5 Earth days (Artemis III surface stay)
    zone_radius_m: float = 300.0
    profile: str = "balanced"
    divert_radius_m: float = 3000.0
    n_backups: int = 5
    step_hours: float = 2.0


def mission_sky(lat: float, lon: float, p: EvalParams) -> tuple[dict, int]:
    times = time_range(p.landing_time, p.mission_hours, p.step_hours)
    from .conditions import visibility

    return visibility(lat, lon, times), len(times)


def evaluate_site(lat: float, lon: float, p: EvalParams, with_backups: bool = True, terrain: TerrainStats | None = None) -> dict:
    terrain = terrain or analyze_zone(lat, lon, p.zone_radius_m)
    sky, steps = mission_sky(lat, lon, p)
    hz = sky["horizon"]
    factors = {
        # Keep the rule score independent of the optional auto-labelled hazard model.
        "ground": ground_factor(terrain, None),
        "sunlight": sunlight_factor(sky, hz, steps),
        "earth": earth_factor(sky, hz, steps),
        "craters": crater_factor(lat, lon),
        "science": science_factor(lat),
    }
    backups = None
    if with_backups:
        from .backups import plan_backups

        backups = plan_backups(lat, lon, p)
        factors["backups"] = backups["factor"]
    overall, verdict, reasons = combine(factors, p.profile)
    out = {
        "lat": lat,
        "lon": lon,
        "landing_time": to_utc(p.landing_time).isoformat(),
        "profile": p.profile,
        "overall": round(overall, 1),
        "verdict": verdict,
        "verdict_reasons": reasons,
        "factors": [f.to_dict() for f in factors.values()],
        "terrain": terrain.to_dict(),
        "horizon": hz.to_dict(),
        "score_basis": "Heuristic planning rules applied to measured terrain and computed visibility; not a measured landing-success probability.",
        "unassessed_factors": [f.key for f in factors.values() if f.score is None],
    }
    if backups is not None:
        out["backups"] = backups["sites"]
        out["divert"] = backups["divert"]
    return out


def combine(factors: dict[str, Factor], profile: str) -> tuple[float, str, list[str]]:
    w = PROFILES.get(profile, PROFILES["balanced"])
    keys = [k for k in w if k in factors and factors[k].score is not None]
    missing = [FACTOR_LABELS[k] for k in w if k in factors and factors[k].score is None]
    incomplete = [f"Not assessed: {', '.join(missing)}."] if missing else []
    if not keys:
        return 0.0, "no-go", incomplete or ["No assessed factors."]
    tot = sum(w[k] for k in keys)
    overall = sum(w[k] * factors[k].score for k in keys) / tot
    fails = [factors[k].summary for k in keys if factors[k].hard_fail]
    # minimum acceptable score per risk factor for a GO (science is value, not risk)
    floors = {"ground": 60, "sunlight": 40, "earth": 40, "craters": 40, "backups": 45}
    weak = [FACTOR_LABELS[k] for k in keys if k in floors and factors[k].score < floors[k] and not factors[k].hard_fail]
    if fails:
        overall = min(overall, 35.0)
        return overall, "no-go", fails + incomplete
    if overall >= 70 and not weak:
        return (overall, "caution", incomplete) if missing else (overall, "go", ["All factors within limits."])
    if overall >= 45:
        return overall, "caution", ([f"Weak: {', '.join(weak)}"] if weak else ["Some factors are marginal."]) + incomplete
    return overall, "no-go", ([f"Weak: {', '.join(weak)}"] if weak else ["Overall suitability too low."]) + incomplete


# ---------------------------------------------------------------- timeline


def timeline(lat: float, lon: float, start: datetime, days: float, step_hours: float, mission_hours: float) -> dict:
    """Landing-time quality over a period → best landing windows.

    Terrain doesn't change with time, so the time score uses lighting and Earth contact only.
    """
    from .conditions import visibility

    total_h = days * 24 + mission_hours
    times = time_range(start, total_h, step_hours)
    sky = visibility(lat, lon, times)
    hz = sky["horizon"]
    sun_clear, earth_clear = sky["sun_clear"], sky["earth_clear"]
    lit = sky["lit"].astype(float)
    vis = sky["earth_ok"].astype(float)
    m = max(1, int(mission_hours / step_hours))
    n = int(days * 24 / step_hours) + 1
    csl = np.concatenate([[0], np.cumsum(lit)])
    cse = np.concatenate([[0], np.cumsum(vis)])
    idx = np.arange(n)
    lit_frac = (csl[idx + m] - csl[idx]) / m
    vis_frac = (cse[idx + m] - cse[idx]) / m
    el = sky["sun_el"][:n]
    angle_q = sun_angle_quality(el) * lit[:n]
    sun_score = 100 * (0.55 * angle_q + 0.45 * lit_frac)
    earth_score = 100 * (0.45 * vis[:n] + 0.55 * vis_frac)
    score = np.where(lit[:n] > 0, 0.6 * sun_score + 0.4 * earth_score, 0.0)

    windows = []
    good = score >= 60
    i = 0
    while i < n:
        if good[i]:
            j = i
            while j + 1 < n and good[j + 1]:
                j += 1
            k = i + int(np.argmax(score[i : j + 1]))
            windows.append(
                {
                    "start": times[i].isoformat(),
                    "end": (times[j] + timedelta(hours=step_hours)).isoformat(),
                    "best": times[k].isoformat(),
                    "best_score": round(float(score[k]), 1),
                }
            )
            i = j + 1
        else:
            i += 1
    windows.sort(key=lambda w: -w["best_score"])
    return {
        "lat": lat,
        "lon": lon,
        "step_hours": step_hours,
        "mission_hours": mission_hours,
        "times": [t.isoformat() for t in times[:n]],
        "sun_elevation": np.round(sky["sun_el"][:n], 2).tolist(),
        "sun_clearance": np.round(sun_clear[:n], 2).tolist(),
        "earth_elevation": np.round(sky["earth_el"][:n], 2).tolist(),
        "earth_clearance": np.round(earth_clear[:n], 2).tolist(),
        "lit_fraction": np.round(lit_frac, 3).tolist(),
        "earth_fraction": np.round(vis_frac, 3).tolist(),
        "score": np.round(score, 1).tolist(),
        "windows": windows[:8],
    }
