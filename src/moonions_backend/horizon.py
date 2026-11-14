"""Terrain horizon profiles.

Near the lunar poles the Sun and Earth skim the horizon, so whether a site is lit or has
radio contact depends on surrounding mountains, not just on geometry. We ray-march the DEM
along each azimuth and keep the maximum elevation angle of the terrain, including the
curvature of the Moon.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import MOON_RADIUS_M
from .dem import destination, get_dem

N_AZ = 180  # 2 degree bins
OBSERVER_HEIGHT_M = 3.0  # lander camera / antenna height


def _distances(max_km: float) -> np.ndarray:
    # dense close in, sparser far out: 100 m steps → ~2 km steps
    near = np.arange(150.0, 5_000.0, 150.0)
    mid = np.arange(5_000.0, 40_000.0, 600.0)
    far = np.arange(40_000.0, max_km * 1000.0, 2_000.0)
    return np.concatenate([near, mid, far])


@lru_cache(maxsize=4096)
def _profile_cached(lat_q: float, lon_q: float, max_km: float, n_az: int = N_AZ) -> np.ndarray:
    dem = get_dem()
    az = np.arange(n_az) * (360.0 / n_az)
    d = _distances(max_km)
    AZ, D = np.meshgrid(az, d, indexing="ij")
    lat, lon = destination(lat_q, lon_q, AZ, D)
    h = dem.elevation(lat, lon)
    h0 = float(dem.elevation(np.array(lat_q), np.array(lon_q))) + OBSERVER_HEIGHT_M
    drop = D**2 / (2 * MOON_RADIUS_M)
    ang = np.degrees(np.arctan2(h - h0 - drop, D))
    return ang.max(axis=1)


class Horizon:
    def __init__(self, lat: float, lon: float, max_km: float | None = None, n_az: int = N_AZ):
        # Far terrain only matters where the sun is low (high latitudes)
        if max_km is None:
            max_km = 150.0 if abs(lat) > 70 else 60.0
        self.lat, self.lon = lat, lon
        self.n_az = n_az
        self.profile = _profile_cached(round(lat, 4), round(lon, 4), max_km, n_az)

    def at(self, az_deg) -> np.ndarray:
        """Terrain horizon elevation (deg) for the given azimuth(s), linear interpolation."""
        n = self.n_az
        step = 360.0 / n
        x = np.mod(np.asarray(az_deg), 360.0) / step
        i0 = np.floor(x).astype(int) % n
        i1 = (i0 + 1) % n
        f = x - np.floor(x)
        return self.profile[i0] * (1 - f) + self.profile[i1] * f

    def to_dict(self) -> dict:
        return {"azimuth_step_deg": 360.0 / self.n_az, "elevation_deg": np.round(self.profile, 2).tolist()}
