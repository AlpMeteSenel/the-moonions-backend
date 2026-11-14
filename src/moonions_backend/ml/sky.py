"""Vectorised Sun/Earth geometry for many sites x many times.

`ephemeris.sky_from_site` is exact but costs ~6 ms per site. For scanning thousands of sites
across many dates we compute the Sun and Earth position vectors in the lunar ME frame once per
time step, then derive local elevation/azimuth for every site with array maths (includes the
observer's offset from the Moon's centre, i.e. topocentric parallax).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..config import MOON_RADIUS_M
from ..ephemeris import get_ephem, to_utc


def body_vectors_km(times: list[datetime]) -> dict[str, np.ndarray]:
    """Sun & Earth position relative to the Moon's centre in MOON_ME frame, km, shape (T, 3)."""
    e = get_ephem()
    t = e.ts.from_datetimes([to_utc(x) for x in times])
    out = {}
    for name, body in (("sun", e.sun), ("earth", e.earth)):
        v = (body - e.moon).at(t).frame_xyz(e.frame).km  # (3, T)
        out[name] = np.asarray(v).T
    return out


def local_frames(lat: np.ndarray, lon: np.ndarray):
    la, lo = np.radians(lat), np.radians(lon)
    up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    return up, east, north


def sky_many(lat, lon, vecs: dict[str, np.ndarray], height_m=None) -> dict[str, np.ndarray]:
    """Elevation / azimuth (deg) of Sun and Earth for N sites x T times → arrays (N, T)."""
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    h = np.zeros_like(lat) if height_m is None else np.asarray(height_m, dtype=np.float64)
    up, east, north = local_frames(lat, lon)
    pos_km = up * ((MOON_RADIUS_M + h) / 1000.0)[:, None]  # (N, 3)
    out = {}
    dirs = {}
    for name, v in vecs.items():
        d = v[None, :, :] - pos_km[:, None, :]  # (N, T, 3)
        dist = np.linalg.norm(d, axis=-1)
        d = d / dist[..., None]
        dirs[name] = d
        u = np.einsum("ntk,nk->nt", d, up)
        e = np.einsum("ntk,nk->nt", d, east)
        n = np.einsum("ntk,nk->nt", d, north)
        out[f"{name}_el"] = np.degrees(np.arcsin(np.clip(u, -1, 1)))
        out[f"{name}_az"] = np.degrees(np.arctan2(e, n)) % 360.0
        out[f"{name}_dist_km"] = dist
    if "sun" in dirs and "earth" in dirs:
        # angular separation of the Sun and Earth seen from the site (for eclipses by Earth)
        cos = np.clip(np.einsum("ntk,ntk->nt", dirs["sun"], dirs["earth"]), -1, 1)
        out["sun_earth_sep_deg"] = np.degrees(np.arccos(cos))
    return out
