"""Landing-zone terrain analysis: slope, roughness, unsafe-ground fraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dem import Source, get_dem

SAFE_SLOPE_DEG = 8.0  # typical lander tip-over / leg-stroke limit range 8-15 deg
MAX_SLOPE_DEG = 15.0


@dataclass
class TerrainStats:
    slope_deg: float  # mean slope at the landing point (zone radius)
    slope_p90_deg: float
    roughness_m: float  # RMS height residual after removing a plane
    unsafe_fraction: float  # share of zone steeper than SAFE_SLOPE_DEG
    relief_m: float  # max-min height within zone
    elevation_m: float
    resolution_m: float
    source: str

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def slope_grid(h: np.ndarray, res_m: float) -> np.ndarray:
    gy, gx = np.gradient(h.astype(np.float64), res_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def analyze_zone(lat: float, lon: float, zone_radius_m: float = 300.0, source: Source | None = None) -> TerrainStats:
    dem = get_dem()
    src = source or dem.source_for(lat)
    res = max(src.resolution_m, 20.0)
    # Need at least a 5x5 neighbourhood to get a meaningful slope estimate
    half = max(zone_radius_m, 2.5 * res)
    east, north, h, src = dem.local_grid(lat, lon, half, res, source=src)
    s = slope_grid(h, res)
    E, N = np.meshgrid(east, north)
    inside = np.hypot(E, N) <= half + 1e-6
    hz = h[inside]
    A = np.c_[E[inside], N[inside], np.ones(inside.sum())]
    coef, *_ = np.linalg.lstsq(A, hz, rcond=None)
    resid = hz - A @ coef
    sz = s[inside]
    return TerrainStats(
        slope_deg=float(np.mean(sz)),
        slope_p90_deg=float(np.percentile(sz, 90)),
        roughness_m=float(np.sqrt(np.mean(resid**2))),
        unsafe_fraction=float(np.mean(sz > SAFE_SLOPE_DEG)),
        relief_m=float(hz.max() - hz.min()),
        elevation_m=float(h[h.shape[0] // 2, h.shape[1] // 2]),
        resolution_m=float(res),
        source=src.name,
    )


def terrain_patch(lat: float, lon: float, half_size_m: float = 4000.0, max_cells: int = 161) -> dict:
    """Heightfield + slope for the 3D site view."""
    dem = get_dem()
    src = dem.source_for(lat)
    res = max(src.resolution_m, 2 * half_size_m / (max_cells - 1))
    east, north, h, src = dem.local_grid(lat, lon, half_size_m, res, source=src)
    s = slope_grid(h, res)
    return {
        "lat": lat,
        "lon": lon,
        "half_size_m": half_size_m,
        "resolution_m": res,
        "source": src.name,
        "source_resolution_m": src.resolution_m,
        "size": int(h.shape[0]),
        "min_m": float(h.min()),
        "max_m": float(h.max()),
        "heights": np.round(h - h[h.shape[0] // 2, h.shape[1] // 2], 1).ravel().tolist(),
        "slopes": np.round(s, 1).ravel().tolist(),
    }
