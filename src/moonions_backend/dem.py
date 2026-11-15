"""Digital elevation model access.

Provides a single interface over several LOLA products and always uses the most
detailed one available for a location:

* LOLA polar stereographic DEMs, 80 m/px, |lat| >= 80
* LOLA GDR global simple-cylindrical DEM, 64 px/deg (~474 m/px)
* NASA SVS CGI Moon Kit displacement map ldem_16.tif, 16 px/deg (~1.9 km/px) fallback

Heights are metres relative to the 1737.4 km reference sphere.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import MOON_RADIUS_M, RAW

log = logging.getLogger(__name__)


def _bilinear(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray, wrap_cols: bool, scale: float) -> np.ndarray:
    h, w = arr.shape
    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    fr = rows - r0
    fc = cols - c0
    r0c = np.clip(r0, 0, h - 1)
    r1c = np.clip(r0 + 1, 0, h - 1)
    if wrap_cols:
        c0c = np.mod(c0, w)
        c1c = np.mod(c0 + 1, w)
    else:
        c0c = np.clip(c0, 0, w - 1)
        c1c = np.clip(c0 + 1, 0, w - 1)
    v00 = arr[r0c, c0c].astype(np.float32)
    v01 = arr[r0c, c1c].astype(np.float32)
    v10 = arr[r1c, c0c].astype(np.float32)
    v11 = arr[r1c, c1c].astype(np.float32)
    top = v00 * (1 - fc) + v01 * fc
    bot = v10 * (1 - fc) + v11 * fc
    return (top * (1 - fr) + bot * fr) * scale


def _read_label(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = re.match(r"\s*([A-Z_^]+)\s*=\s*(.+?)\s*$", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2).split("<")[0].strip().strip('"')
    return out


def _img_array(path: Path, meta: dict[str, str]) -> np.memmap:
    """Reject incomplete/mismatched PDS products before reading terrain values."""
    if meta["SAMPLE_TYPE"] != "LSB_INTEGER" or int(meta["SAMPLE_BITS"]) != 16:
        raise ValueError(f"Unsupported PDS sample encoding: {path.name}")
    shape = int(meta["LINES"]), int(meta["LINE_SAMPLES"])
    expected = shape[0] * shape[1] * 2
    if path.stat().st_size != expected:
        raise ValueError(f"{path.name}: expected {expected} bytes from PDS label, got {path.stat().st_size}")
    return np.memmap(path, dtype="<i2", mode="r", shape=shape)


@dataclass
class Source:
    name: str
    resolution_m: float

    def covers(self, lat: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class CylindricalImg(Source):
    """LOLA GDR simple-cylindrical PDS .img (int16, 0..360 E, 90N at top)."""

    def __init__(self, img: Path, lbl: Path):
        meta = _read_label(lbl)
        self.lines = int(meta["LINES"])
        self.samples = int(meta["LINE_SAMPLES"])
        self.ppd = float(meta["MAP_RESOLUTION"])
        self.scale = float(meta["SCALING_FACTOR"])
        self.line_off = float(meta["LINE_PROJECTION_OFFSET"])
        self.sample_off = float(meta["SAMPLE_PROJECTION_OFFSET"])
        self.center_lon = float(meta["CENTER_LONGITUDE"])
        self.arr = _img_array(img, meta)
        super().__init__(name=f"LOLA LDEM_{int(self.ppd)}", resolution_m=float(meta["MAP_SCALE"]))

    def covers(self, lat):
        return np.ones_like(np.asarray(lat), dtype=bool)

    def sample(self, lat, lon):
        lon360 = np.mod(lon, 360.0)
        cols = self.sample_off + (lon360 - self.center_lon) * self.ppd
        rows = self.line_off - np.asarray(lat) * self.ppd
        return _bilinear(self.arr, rows, cols, True, self.scale)


class CylindricalTif(Source):
    """NASA SVS ldem_*.tif float km, lon -180 at left, 90N at top."""

    def __init__(self, path: Path):
        import tifffile

        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        self.arr = arr
        h, w = arr.shape
        self.ppd = w / 360.0
        # SVS maps are in km; convert to metres
        self.scale = 1000.0
        super().__init__(name=f"SVS LDEM_{int(self.ppd)}", resolution_m=2 * np.pi * MOON_RADIUS_M / w)

    def covers(self, lat):
        return np.ones_like(np.asarray(lat), dtype=bool)

    def sample(self, lat, lon):
        lon180 = (np.asarray(lon) + 180.0) % 360.0 - 180.0
        cols = (lon180 + 180.0) * self.ppd - 0.5
        rows = (90.0 - np.asarray(lat)) * self.ppd - 0.5
        return _bilinear(self.arr, rows, cols, True, self.scale)


class PolarImg(Source):
    """LOLA polar stereographic PDS .img (int16)."""

    def __init__(self, img: Path, lbl: Path):
        meta = _read_label(lbl)
        self.lines = int(meta["LINES"])
        self.samples = int(meta["LINE_SAMPLES"])
        self.scale = float(meta["SCALING_FACTOR"])
        self.map_scale = float(meta["MAP_SCALE"])
        self.line_off = float(meta["LINE_PROJECTION_OFFSET"])
        self.sample_off = float(meta["SAMPLE_PROJECTION_OFFSET"])
        self.center_lat = float(meta["CENTER_LATITUDE"])
        self.center_lon = float(meta["CENTER_LONGITUDE"])
        self.south = self.center_lat < 0
        self.min_abs_lat = float(meta["MAXIMUM_LATITUDE"] if self.south else meta["MINIMUM_LATITUDE"])
        self.min_abs_lat = abs(self.min_abs_lat)
        self.arr = _img_array(img, meta)
        pole = "S" if self.south else "N"
        super().__init__(name=f"LOLA polar {pole} {int(self.map_scale)} m", resolution_m=self.map_scale)

    def covers(self, lat):
        lat = np.asarray(lat)
        # keep a margin from the corner-cut edge of the map
        return (lat <= -(self.min_abs_lat + 0.3)) if self.south else (lat >= self.min_abs_lat + 0.3)

    def project(self, lat, lon):
        lat_r = np.radians(np.asarray(lat))
        dlon = np.radians(np.asarray(lon) - self.center_lon)
        R = MOON_RADIUS_M
        if self.south:
            rho = 2 * R * np.tan(np.pi / 4 + lat_r / 2)
            x = rho * np.sin(dlon)
            y = rho * np.cos(dlon)
        else:
            rho = 2 * R * np.tan(np.pi / 4 - lat_r / 2)
            x = rho * np.sin(dlon)
            y = -rho * np.cos(dlon)
        return x, y

    def sample(self, lat, lon):
        x, y = self.project(lat, lon)
        cols = self.sample_off + x / self.map_scale
        rows = self.line_off - y / self.map_scale
        return _bilinear(self.arr, rows, cols, False, self.scale)


class DEM:
    """Best-available elevation lookup over all loaded sources."""

    def __init__(self, raw: Path = RAW):
        self.polar: list[PolarImg] = []
        self.global_src: Source | None = None
        for pole in ("80s", "80n"):
            img, lbl = raw / f"ldem_{pole}_80m.img", raw / f"ldem_{pole}_80m.lbl"
            if img.exists() and lbl.exists():
                try:
                    self.polar.append(PolarImg(img, lbl))
                except Exception as exc:  # partial download etc.
                    log.warning("polar DEM %s unusable: %s", img, exc)
        img, lbl = raw / "ldem_64.img", raw / "ldem_64.lbl"
        if img.exists() and lbl.exists():
            self.global_src = CylindricalImg(img, lbl)
        elif (raw / "ldem_16.tif").exists():
            self.global_src = CylindricalTif(raw / "ldem_16.tif")
        if self.global_src is None:
            raise FileNotFoundError("No DEM found. Run scripts/download_data.sh first.")

    @property
    def sources(self) -> list[Source]:
        return [*self.polar, *([self.global_src] if self.global_src else [])]

    def source_for(self, lat: float) -> Source:
        for p in self.polar:
            if bool(p.covers(np.array(lat))):
                return p
        assert self.global_src is not None
        return self.global_src

    def elevation(self, lat, lon, source: Source | None = None) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if source is not None:
            return source.sample(lat, lon)
        assert self.global_src is not None
        out = self.global_src.sample(lat, lon)
        for p in self.polar:
            m = p.covers(lat)
            if np.any(m):
                out = np.where(m, p.sample(lat, lon), out)
        return out

    def local_grid(self, lat0: float, lon0: float, half_size_m: float, res_m: float | None = None, source: Source | None = None):
        """Height grid in a local east/north tangent frame centred on (lat0, lon0).

        Returns (east_m, north_m, heights_m[ny, nx], source). Row 0 is the northern edge.
        """
        src = source or self.source_for(lat0)
        res = res_m or max(src.resolution_m, 20.0)
        n = int(round(2 * half_size_m / res)) + 1
        axis = np.linspace(-half_size_m, half_size_m, n)
        east, north = np.meshgrid(axis, axis[::-1])
        lat, lon = offset_latlon(lat0, lon0, east, north)
        h = self.elevation(lat, lon, source=src)
        return axis, axis[::-1], h, src


def offset_latlon(lat0, lon0, east_m, north_m):
    """Azimuthal-equidistant inverse: local east/north offsets → lat/lon (degrees)."""
    d = np.hypot(east_m, north_m) / MOON_RADIUS_M
    az = np.arctan2(east_m, north_m)
    return destination(lat0, lon0, np.degrees(az), d * MOON_RADIUS_M)


def destination(lat0, lon0, bearing_deg, dist_m):
    phi1 = np.radians(lat0)
    lam1 = np.radians(lon0)
    th = np.radians(bearing_deg)
    d = np.asarray(dist_m) / MOON_RADIUS_M
    sin_phi2 = np.sin(phi1) * np.cos(d) + np.cos(phi1) * np.sin(d) * np.cos(th)
    phi2 = np.arcsin(np.clip(sin_phi2, -1, 1))
    lam2 = lam1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(phi1), np.cos(d) - np.sin(phi1) * sin_phi2)
    lon2 = (np.degrees(lam2) + 180.0) % 360.0 - 180.0
    return np.degrees(phi2), lon2


def distance_bearing(lat1, lon1, lat2, lon2):
    """Great-circle distance (m) and initial bearing (deg) from point 1 to point 2."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * MOON_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    b = np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))
    return d, (b + 360) % 360


@lru_cache(maxsize=1)
def get_dem() -> DEM:
    dem = DEM()
    log.info("DEM sources: %s", [s.name for s in dem.sources])
    return dem
