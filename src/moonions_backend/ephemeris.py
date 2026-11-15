"""Sun and Earth positions as seen from points on the Moon.

Uses JPL DE421 planetary ephemeris and the NAIF lunar principal-axis / mean-Earth frame
(MOON_ME_DE421), so libration and the real lunar orientation are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from skyfield.api import Loader
from skyfield.planetarylib import PlanetaryConstants

from .config import RAW


@dataclass
class Ephem:
    ts: object
    eph: object
    moon: object
    sun: object
    earth: object
    pc: PlanetaryConstants
    frame: object


@lru_cache(maxsize=1)
def get_ephem() -> Ephem:
    load = Loader(str(RAW), verbose=False)
    ts = load.timescale(builtin=True)
    eph = load("de421.bsp")
    pc = PlanetaryConstants()
    pc.read_text(load("moon_080317.tf"))
    pc.read_text(load("pck00011.tpc"))
    pc.read_binary(load("moon_pa_de421_1900-2050.bpc"))
    frame = pc.build_frame_named("MOON_ME_DE421")
    return Ephem(ts, eph, eph["moon"], eph["sun"], eph["earth"], pc, frame)


def to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def time_range(start: datetime, hours: float, step_h: float) -> list[datetime]:
    start = to_utc(start)
    n = int(hours / step_h) + 1
    return [start + timedelta(hours=i * step_h) for i in range(n)]


def sky_from_site(lat: float, lon: float, times: list[datetime]) -> dict[str, np.ndarray]:
    """Sun & Earth elevation/azimuth (degrees, azimuth clockwise from north) over a list of times.

    Elevation is relative to the local *geometric* horizon (a smooth sphere); compare with
    `horizon.Horizon` to account for terrain.
    """
    e = get_ephem()
    t = e.ts.from_datetimes([to_utc(x) for x in times])
    place = e.moon + e.pc.build_latlon_degrees(e.frame, lat, lon)
    out: dict[str, np.ndarray] = {}
    for name, body in (("sun", e.sun), ("earth", e.earth)):
        app = place.at(t).observe(body).apparent()
        alt, az, _ = app.altaz()
        out[f"{name}_el"] = np.atleast_1d(alt.degrees)
        out[f"{name}_az"] = np.atleast_1d(az.degrees)
    return out


def _sub_point(body_name: str, when: datetime) -> tuple[float, float]:
    e = get_ephem()
    t = e.ts.from_datetime(to_utc(when))
    body = e.sun if body_name == "sun" else e.earth
    vec = (body - e.moon).at(t)
    x, y, z = vec.frame_xyz(e.frame).au
    r = np.sqrt(x * x + y * y + z * z)
    return float(np.degrees(np.arcsin(z / r))), float(np.degrees(np.arctan2(y, x)))


def sun_and_earth_points(when: datetime) -> dict[str, float]:
    """Sub-solar and sub-Earth selenographic points (for globe lighting)."""
    slat, slon = _sub_point("sun", when)
    elat, elon = _sub_point("earth", when)
    return {"sun_lat": slat, "sun_lon": slon, "earth_lat": elat, "earth_lon": elon}
