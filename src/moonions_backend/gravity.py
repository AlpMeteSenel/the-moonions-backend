"""Lunar gravity: geoid (equipotential surface) and local gravity from NASA GRAIL.

Slopes that matter for a lander are measured against the *local horizontal*, i.e. the
equipotential surface the plumb line is perpendicular to — not against a perfect sphere.
Mass anomalies (mascons) tilt the plumb line ("deflection of the vertical"). We compute the
geoid N(lat, lon) from the GRAIL gravity model GRGM1200B (Goossens et al. 2020), truncated at
degree 600 where the model is data-driven (~9 km resolution), including the Moon's rotation.
Terrain height above the equipotential is then h - N, and slopes use that.

  uv run python -m moonions_backend.gravity --build      # one-off, downloads GRGM1200B
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np

from .config import CACHE, MOON_RADIUS_M

GEOID = CACHE / "geoid_grgm1200b_l600.npz"
GM_MOON = 4.9028001218467e12  # m^3/s^2 (GRGM1200B)
OMEGA_MOON = 2.6616995e-6  # rad/s


def build(lmax: int = 600) -> None:
    import pyshtools as pysh

    clm = pysh.datasets.Moon.GRGM1200B(lmax=lmax)
    r = MOON_RADIUS_M
    potref = clm.gm / r + 0.5 * (OMEGA_MOON * r) ** 2 * (2 / 3)  # mean potential on the reference sphere
    geoid = clm.geoid(potref=potref, r=r, omega=OMEGA_MOON, lmax=lmax, grid="DH2", extend=True)
    grid = geoid.geoid
    data = grid.to_array().astype(np.float32)  # rows: lat 90 → -90, cols: lon 0 → 360
    lats = grid.lats()
    lons = grid.lons()
    np.savez_compressed(GEOID, N=data, lats=lats.astype(np.float64), lons=lons.astype(np.float64), lmax=lmax)
    print("geoid saved", GEOID, data.shape, f"range {data.min():.0f}..{data.max():.0f} m")


@lru_cache(maxsize=1)
def _geoid():
    if not GEOID.exists():
        return None
    d = np.load(GEOID)
    return d["N"], d["lats"], d["lons"]


def available() -> bool:
    return GEOID.exists()


def geoid_height(lat, lon) -> np.ndarray:
    """Geoid height N (m) above the 1737.4 km sphere; zeros if the model isn't built."""
    g = _geoid()
    lat = np.asarray(lat, dtype=np.float64)
    if g is None:
        return np.zeros_like(lat)
    N, lats, lons = g
    dlat = lats[0] - lats[1]
    dlon = lons[1] - lons[0]
    rows = (lats[0] - lat) / dlat
    cols = np.mod(np.asarray(lon, dtype=np.float64), 360.0) / dlon
    r0 = np.clip(np.floor(rows).astype(int), 0, N.shape[0] - 2)
    c0 = np.floor(cols).astype(int)
    fr = np.clip(rows - r0, 0, 1)
    fc = cols - c0
    c0 %= N.shape[1] - 1  # extended grid repeats lon 360 = 0
    c1 = c0 + 1
    top = N[r0, c0] * (1 - fc) + N[r0, c1] * fc
    bot = N[r0 + 1, c0] * (1 - fc) + N[r0 + 1, c1] * fc
    return top * (1 - fr) + bot * fr


def surface_gravity(lat, elevation_m) -> np.ndarray:
    """Magnitude of gravity (m/s^2) at the surface: GM/r^2 minus the (tiny) centrifugal part."""
    r = MOON_RADIUS_M + np.asarray(elevation_m, dtype=np.float64)
    return GM_MOON / r**2 - OMEGA_MOON**2 * r * np.cos(np.radians(lat)) ** 2


def deflection_deg(lat, lon, step_m: float = 2000.0) -> np.ndarray:
    """Deflection of the vertical (deg): tilt of the true horizontal vs the sphere's tangent plane."""
    from .dem import destination

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    ln, lo_n = destination(lat, lon, 0.0, step_m)
    le, lo_e = destination(lat, lon, 90.0, step_m)
    n0 = geoid_height(lat, lon)
    gn = (geoid_height(ln, lo_n) - n0) / step_m
    ge = (geoid_height(le, lo_e) - n0) / step_m
    return np.degrees(np.arctan(np.hypot(gn, ge)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lmax", type=int, default=600)
    a = ap.parse_args()
    if a.build:
        build(a.lmax)
