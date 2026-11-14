"""Robbins (2018) global lunar crater database: density and proximity queries."""

from __future__ import annotations

import logging
import zipfile
from functools import lru_cache

import numpy as np

from .config import CACHE, MOON_RADIUS_M, RAW

log = logging.getLogger(__name__)
NPZ = CACHE / "craters.npz"


def _unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def build_cache() -> bool:
    """Convert the Robbins CSV (inside the zip, or extracted) to a compact npz."""
    import pandas as pd

    src = None
    for cand in [RAW / "lunar_crater_database_robbins_2018.csv", *RAW.glob("robbins*/*.csv")]:
        if cand.exists():
            src = cand
    zpath = RAW / "robbins_2018.zip"
    if src is None and zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            names = sorted((i.filename for i in z.infolist() if i.filename.lower().endswith(".csv")), key=lambda n: -z.getinfo(n).file_size)
            if not names:
                return False
            with z.open(names[0]) as f:
                df = pd.read_csv(f)
    elif src is not None:
        df = pd.read_csv(src)
    else:
        return False
    cols = {c.upper(): c for c in df.columns}
    lat = df[cols["LAT_CIRC_IMG"]].to_numpy(np.float32)
    lon = df[cols["LON_CIRC_IMG"]].to_numpy(np.float32)
    diam = df[cols["DIAM_CIRC_IMG"]].to_numpy(np.float32)  # km
    lon = ((lon + 180) % 360 - 180).astype(np.float32)
    np.savez_compressed(NPZ, lat=lat, lon=lon, diam_km=diam)
    log.info("crater cache built: %d craters", len(lat))
    return True


class CraterDB:
    def __init__(self):
        from scipy.spatial import cKDTree

        d = np.load(NPZ)
        self.lat, self.lon, self.diam_km = d["lat"], d["lon"], d["diam_km"]
        self.tree = cKDTree(_unit(self.lat, self.lon))

    def around(self, lat: float, lon: float, radius_m: float) -> dict:
        chord = 2 * np.sin(radius_m / MOON_RADIUS_M / 2)
        idx = self.tree.query_ball_point(_unit(lat, lon)[0], chord)
        area_km2 = np.pi * (radius_m / 1000) ** 2
        if not idx:
            return {"count": 0, "density_per_1000km2": 0.0, "largest_km": 0.0, "inside_crater": False, "nearest_rim_km": None}
        idx = np.asarray(idx)
        # distance from site to crater centre
        u0 = _unit(lat, lon)[0]
        dots = np.clip(_unit(self.lat[idx], self.lon[idx]) @ u0, -1, 1)
        dist_km = np.arccos(dots) * MOON_RADIUS_M / 1000
        rim_dist = np.abs(dist_km - self.diam_km[idx] / 2)
        inside = dist_km < self.diam_km[idx] / 2
        # being near a rim (steep walls, ejecta boulders) matters most for small/medium craters
        return {
            "count": int(len(idx)),
            "density_per_1000km2": float(len(idx) / area_km2 * 1000),
            "largest_km": float(self.diam_km[idx].max()),
            "inside_crater": bool(inside.any()),
            "nearest_rim_km": float(rim_dist.min()),
        }


@lru_cache(maxsize=1)
def get_craters() -> CraterDB | None:
    if not NPZ.exists():
        try:
            if not build_cache():
                return None
        except Exception as exc:
            log.warning("crater DB unavailable: %s", exc)
            return None
    return CraterDB()
