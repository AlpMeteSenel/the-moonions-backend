"""Surface temperature at a site over time: a two-node lunar regolith energy balance.

    C dT/dt = (1 - A) S(d) cos(i) f_vis + F_floor + k (T_deep - T) - eps sigma T^4

* Sunlight on the local slope (LOLA slope and aspect), times the visible share of the solar disk
  over the terrain skyline (so ridges cast real shadows and permanently shadowed ground stays cold).
* S scales with the actual Sun distance; albedo A and emissivity eps are Diviner-era averages.
* C is the effective heat capacity of the thermal skin (thermal inertia ~ 55 SI over one lunation),
  k couples it to a deeper layer at T_deep, the site's mean equilibrium temperature over the
  preceding lunar day; F_floor is geothermal heat plus scattered light/IR from nearby terrain.
* Spun up over the 30 days before the window, so the start is not arbitrary.

Tuned to Diviner's typical values: equatorial noon ~390 K, equatorial pre-dawn ~95-100 K,
polar permanently shadowed floors ~40-60 K.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

SOLAR_CONSTANT = 1361.0  # W/m² at 1 AU
AU_KM = 149_597_870.7
ALBEDO = 0.12
EMISSIVITY = 0.95
SIGMA = 5.670374419e-8
HEAT_CAPACITY = 30_000.0  # J m⁻² K⁻¹ (effective, diurnal skin)
COUPLING = 0.036  # W m⁻² K⁻¹ to the deeper layer
FLOOR_FLUX = 0.3  # W/m²: geothermal (~0.016) + scattered sunlight / IR from surrounding terrain
STEP_S = 600.0  # internal time step (stable for these constants)


def slope_aspect(lat: float, lon: float, d_m: float = 120.0) -> tuple[float, float]:
    """Local slope (deg) and aspect (deg, direction the ground faces, from north) from the DEM."""
    from .config import MOON_RADIUS_M
    from .dem import get_dem

    dem = get_dem()
    dlat = np.degrees(d_m / MOON_RADIUS_M)
    dlon = dlat / max(np.cos(np.radians(lat)), 1e-3)
    pts_lat = np.array([lat, lat, lat + dlat, lat - dlat])
    pts_lon = np.array([lon + dlon, lon - dlon, lon, lon])
    h = np.asarray(dem.elevation(pts_lat, pts_lon), dtype=float)
    dzdx = (h[0] - h[1]) / (2 * d_m)  # east
    dzdy = (h[2] - h[3]) / (2 * d_m)  # north
    slope = float(np.degrees(np.arctan(np.hypot(dzdx, dzdy))))
    aspect = float(np.degrees(np.arctan2(-dzdx, -dzdy)) % 360)  # downhill direction = facing
    return slope, aspect


def absorbed_flux(sun_el, sun_az, sun_frac, sun_dist_km, slope_deg: float, aspect_deg: float) -> np.ndarray:
    el, az = np.radians(sun_el), np.radians(sun_az)
    s, a = np.radians(slope_deg), np.radians(aspect_deg)
    cos_i = np.sin(el) * np.cos(s) + np.cos(el) * np.sin(s) * np.cos(az - a)
    flux = SOLAR_CONSTANT * (AU_KM / np.asarray(sun_dist_km)) ** 2
    return (1 - ALBEDO) * flux * np.clip(cos_i, 0, None) * np.clip(sun_frac, 0, 1)


def _equilibrium(f_abs: np.ndarray) -> np.ndarray:
    return ((f_abs + FLOOR_FLUX) / (EMISSIVITY * SIGMA)) ** 0.25


def integrate(f_abs: np.ndarray, step_s: float, t0: float, t_deep: float) -> np.ndarray:
    """Explicit integration with internal sub-steps; returns temperature at each forcing sample."""
    sub = max(1, int(np.ceil(step_s / STEP_S)))
    dt = step_s / sub
    out = np.empty(len(f_abs))
    t = t0
    for i, f in enumerate(f_abs):
        for _ in range(sub):
            t += dt * (f + FLOOR_FLUX + COUPLING * (t_deep - t) - EMISSIVITY * SIGMA * t**4) / HEAT_CAPACITY
            t = max(t, 20.0)
        out[i] = t
    return out


def site_temperature(lat: float, lon: float, times: list[datetime], vis: dict, step_h: float, comm_mask_deg: float = 1.0) -> dict:
    """Temperature (K) at each time of `vis` (from conditions.visibility), spun up over the prior lunation."""
    from .conditions import visibility

    slope, aspect = slope_aspect(lat, lon)
    spin_times = [times[0] - timedelta(hours=h) for h in range(30 * 24, 0, -1)]
    pre = visibility(lat, lon, spin_times, comm_mask_deg)
    f_pre = absorbed_flux(pre["sun_el"], pre["sun_az"], pre["sun_frac"], pre["sun_dist_km"], slope, aspect)
    t_deep = float(np.mean(_equilibrium(f_pre)))
    spun = integrate(f_pre, 3600.0, t_deep, t_deep)
    f = absorbed_flux(vis["sun_el"], vis["sun_az"], vis["sun_frac"], vis["sun_dist_km"], slope, aspect)
    temp = integrate(f, step_h * 3600.0, float(spun[-1]), t_deep)
    return {"temp_k": temp, "slope_deg": slope, "aspect_deg": aspect, "deep_k": t_deep}
