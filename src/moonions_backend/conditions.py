"""Sun and Earth conditions at a landing site over time: power and direct-to-Earth windows.

Physically modelled:
* Sun / Earth positions: JPL DE421 + lunar principal-axis frame (real orbits, libration,
  topocentric parallax from the site).
* Terrain skyline: LOLA DEM horizon at 0.5° azimuth resolution, out to 150 km, incl. lunar
  curvature; no atmosphere, so no refraction.
* Finite disks: the Sun (≈0.53°) and Earth (≈1.9° from the Moon) are disks whose apparent size
  follows their actual distance; sunrise/sunset at the pole can take hours as the disk slides
  along a ridge. Power scales with the visible fraction of the solar disk.
* Eclipses: Earth passing in front of the Sun (a lunar eclipse seen from Earth) blocks sunlight.
* Direct-to-Earth comms need Earth's centre a margin above the skyline (antenna mask).
* Battery: vertical solar panels (output ∝ visible sun fraction × cos elevation), constant load.

Rise/set times are found on a 10-minute grid and refined by linear interpolation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .ephemeris import to_utc
from .horizon import Horizon
from .ml.simulate import DEFAULT, energy_soc
from .ml.sky import body_vectors_km, sky_many

SUN_RADIUS_KM = 695_700.0
EARTH_RADIUS_KM = 6_378.1


def angular_radius(radius_km: float, dist_km: np.ndarray) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(radius_km / dist_km, 0, 1)))


def disk_fraction_above(center_clear_deg: np.ndarray, radius_deg: np.ndarray) -> np.ndarray:
    """Fraction of a disk's area above a locally straight skyline (centre clearance in degrees)."""
    u = np.clip(center_clear_deg / radius_deg, -1, 1)
    return (np.arccos(-u) + u * np.sqrt(1 - u * u)) / np.pi


def covered_fraction(sep_deg: np.ndarray, r_small: np.ndarray, r_big: np.ndarray) -> np.ndarray:
    """Fraction of the small disk (Sun) covered by the big disk (Earth) at angular separation sep."""
    d, r, R = np.asarray(sep_deg, float), np.asarray(r_small, float), np.asarray(r_big, float)
    out = np.zeros(np.broadcast(d, r, R).shape)
    full = d <= R - r
    out = np.where(full, 1.0, out)
    part = (d < R + r) & ~full
    if np.any(part):
        dp, rp, Rp = np.broadcast_to(d, out.shape)[part], np.broadcast_to(r, out.shape)[part], np.broadcast_to(R, out.shape)[part]
        a1 = rp**2 * np.arccos(np.clip((dp**2 + rp**2 - Rp**2) / (2 * dp * rp), -1, 1))
        a2 = Rp**2 * np.arccos(np.clip((dp**2 + Rp**2 - rp**2) / (2 * dp * Rp), -1, 1))
        a3 = 0.5 * np.sqrt(np.clip((-dp + rp + Rp) * (dp + rp - Rp) * (dp - rp + Rp) * (dp + rp + Rp), 0, None))
        out[part] = (a1 + a2 - a3) / (np.pi * rp**2)
    return np.clip(out, 0, 1)


def _crossings(t_h: np.ndarray, f: np.ndarray, level: float) -> list[tuple[float, str]]:
    """Times (hours) where f crosses `level`, linearly interpolated, with direction."""
    s = f - level
    out = []
    for i in np.flatnonzero(np.sign(s[:-1]) != np.sign(s[1:])):
        if s[i] == s[i + 1]:
            continue
        x = t_h[i] + (t_h[i + 1] - t_h[i]) * (-s[i]) / (s[i + 1] - s[i])
        out.append((float(x), "rise" if s[i + 1] > s[i] else "set"))
    return out


def _runs(mask: np.ndarray, step_h: float) -> tuple[float, float]:
    """(longest True run, total True) in hours."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * step_h, float(mask.sum() * step_h)


def site_conditions(
    lat: float,
    lon: float,
    start: datetime,
    days: float = 14.0,
    step_min: float = 10.0,
    comm_mask_deg: float = 1.0,
    mission_hours: float = DEFAULT.mission_hours,
) -> dict:
    start = to_utc(start)
    step_h = step_min / 60.0
    n = int(days * 24 / step_h) + 1
    times = [start + timedelta(hours=step_h * i) for i in range(n)]
    hz = Horizon(lat, lon, n_az=720)
    from .dem import get_dem

    elev = float(get_dem().elevation(np.array(lat), np.array(lon)))
    s = {k: v[0] for k, v in sky_many([lat], [lon], body_vectors_km(times), height_m=[elev]).items()}
    sun_r = angular_radius(SUN_RADIUS_KM, s["sun_dist_km"])
    earth_r = angular_radius(EARTH_RADIUS_KM, s["earth_dist_km"])
    sun_clear = s["sun_el"] - hz.at(s["sun_az"])
    earth_clear = s["earth_el"] - hz.at(s["earth_az"])
    above = disk_fraction_above(sun_clear, sun_r)
    eclipsed = covered_fraction(s["sun_earth_sep_deg"], sun_r, earth_r)
    sun_frac = above * (1 - eclipsed)  # visible fraction of the solar disk
    earth_ok = earth_clear >= comm_mask_deg
    lit = sun_frac > 0.5  # "sunlit": most of the solar disk visible

    t_h = np.arange(n) * step_h
    sun_events = _crossings(t_h, sun_frac, 0.5)
    earth_events = _crossings(t_h, earth_clear, comm_mask_deg)

    def next_event(events, kind):
        e = next((x for x, k in events if k == kind and x > 0), None)
        return None if e is None else {"in_hours": round(e, 2), "at": (start + timedelta(hours=e)).isoformat()}

    longest_dark, _ = _runs(~lit, step_h)
    longest_no_earth, _ = _runs(~earth_ok, step_h)
    sun_hours = float(lit.sum() * step_h)
    earth_hours = float(earth_ok.sum() * step_h)

    # battery over the stay (vertical panels track the Sun in azimuth)
    m = min(n, int(mission_hours / step_h) + 1)
    gen_frac = sun_frac[:m] * np.cos(np.radians(np.clip(s["sun_el"][:m], 0, 90)))
    p = DEFAULT
    net = (p.panel_kw * gen_frac - p.load_kw) * step_h
    soc = np.empty(m)
    cur = p.battery_kwh
    for i in range(m):
        cur = min(max(cur + net[i], 0.0), p.battery_kwh)
        soc[i] = cur / p.battery_kwh

    eclipses = []
    for a, b in zip(*[iter(_crossings(t_h, (eclipsed > 0.001).astype(float), 0.5))] * 2):
        k0, k1 = int(a[0] / step_h), int(b[0] / step_h) + 1
        eclipses.append({"start": (start + timedelta(hours=a[0])).isoformat(), "end": (start + timedelta(hours=b[0])).isoformat(), "max_covered": round(float(eclipsed[k0:k1].max()), 3)})
    if eclipsed[0] > 0.001 and not eclipses:
        eclipses.append({"start": start.isoformat(), "end": None, "max_covered": round(float(eclipsed.max()), 3)})

    every = max(1, int(60 / step_min))  # hourly series for charts
    track_every = max(1, int(180 / step_min))
    return {
        "site": {"lat": lat, "lon": lon, "elevation_m": round(elev, 1)},
        "start": start.isoformat(),
        "days": days,
        "comm_mask_deg": comm_mask_deg,
        "now": {
            "sun_elevation_deg": round(float(s["sun_el"][0]), 2),
            "sun_azimuth_deg": round(float(s["sun_az"][0]), 1),
            "sun_above_skyline_deg": round(float(sun_clear[0]), 2),
            "sun_visible_fraction": round(float(sun_frac[0]), 3),
            "earth_elevation_deg": round(float(s["earth_el"][0]), 2),
            "earth_azimuth_deg": round(float(s["earth_az"][0]), 1),
            "earth_above_skyline_deg": round(float(earth_clear[0]), 2),
            "sun_up": bool(lit[0]),
            "earth_up": bool(earth_ok[0]),
            "sun_angular_diameter_deg": round(float(2 * sun_r[0]), 3),
            "earth_angular_diameter_deg": round(float(2 * earth_r[0]), 3),
        },
        "next": {
            "sunset": next_event(sun_events, "set"),
            "sunrise": next_event(sun_events, "rise"),
            "earthset": next_event(earth_events, "set"),
            "earthrise": next_event(earth_events, "rise"),
        },
        "period": {
            "sunlit_hours": round(sun_hours, 1),
            "sunlit_share": round(float(lit.mean()), 3),
            "longest_darkness_hours": round(longest_dark, 1),
            "max_sun_elevation_deg": round(float(s["sun_el"].max()), 2),
            "earth_hours": round(earth_hours, 1),
            "earth_share": round(float(earth_ok.mean()), 3),
            "longest_earth_gap_hours": round(longest_no_earth, 1),
        },
        "stay": {
            "hours": mission_hours,
            "power_ok": bool(soc.min() > 0),
            "lowest_battery": round(float(soc.min()), 3),
            "sunlit_share": round(float(lit[:m].mean()), 3),
            "earth_share": round(float(earth_ok[:m].mean()), 3),
        },
        "eclipses": eclipses,
        "horizon": {"azimuth_step_deg": 0.5, "elevation_deg": np.round(hz.profile, 2).tolist()},
        "series": {
            "times": [t.isoformat() for t in times[::every]],
            "sun_elevation": np.round(s["sun_el"][::every], 2).tolist(),
            "sun_above_skyline": np.round(sun_clear[::every], 2).tolist(),
            "sun_visible_fraction": np.round(sun_frac[::every], 3).tolist(),
            "earth_above_skyline": np.round(earth_clear[::every], 2).tolist(),
            "battery": np.round(soc[::every], 3).tolist(),
        },
        "tracks": {
            "sun": [[round(float(a), 1), round(float(e), 2)] for a, e in zip(s["sun_az"][::track_every], s["sun_el"][::track_every])],
            "earth": [[round(float(a), 1), round(float(e), 2)] for a, e in zip(s["earth_az"][::track_every], s["earth_el"][::track_every])],
        },
    }
