"""Validate the terrain horizon + ephemeris against NASA's LOLA average-illumination maps.

Reference: Mazarico et al. (2011), Icarus 211, 1066-1081; products AVGVISIB_85S_060M_201608
(Sun) and *_EARTH (Earth), fraction of time any part of the disk is visible, from a
ground-level observer, over one lunar precession cycle (~18.6 years).

  uv run python scripts/validate_illumination.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moonions_backend import horizon as hzmod  # noqa: E402
from moonions_backend.config import RAW  # noqa: E402
from moonions_backend.conditions import EARTH_RADIUS_KM, SUN_RADIUS_KM, angular_radius  # noqa: E402
from moonions_backend.dem import PolarImg  # noqa: E402
from moonions_backend.ml.sky import body_vectors_km, sky_many  # noqa: E402

N_SITES, N_TIMES, SEED = 300, 2000, 0
V = RAW / "validation"


def main():
    hzmod.OBSERVER_HEIGHT_M = 0.0  # the NASA maps are for a ground-level observer
    ref_sun = PolarImg(V / "AVGVISIB_85S_060M_201608.IMG", V / "AVGVISIB_85S_060M_201608.LBL")
    ref_earth = PolarImg(V / "AVGVISIB_85S_060M_201608_EARTH.IMG", V / "AVGVISIB_85S_060M_201608_EARTH.LBL")
    rng = np.random.default_rng(SEED)
    colat = np.degrees(np.arccos(1 - rng.uniform(0, 1 - np.cos(np.radians(4.7)), N_SITES)))
    lat, lon = -(90 - colat), rng.uniform(-180, 180, N_SITES)
    t0 = datetime(2010, 1, 1, tzinfo=timezone.utc)
    span_h = 18.6 * 365.25 * 24
    times = sorted(t0 + timedelta(hours=float(h)) for h in rng.uniform(0, span_h, N_TIMES))

    prof = np.stack([hzmod.Horizon(float(a), float(b), n_az=720).profile for a, b in zip(lat, lon)])
    s = sky_many(lat, lon, body_vectors_km(times))
    from moonions_backend.ai import _horizon_at

    out = {}
    for body, R, ref in (("sun", SUN_RADIUS_KM, ref_sun), ("earth", EARTH_RADIUS_KM, ref_earth)):
        clear = s[f"{body}_el"] - _horizon_at(prof, s[f"{body}_az"])
        r = angular_radius(R, s[f"{body}_dist_km"])
        ours = (clear > -r).mean(1)  # any part of the disk above the skyline
        nasa = ref.sample(lat, lon)
        err = ours - nasa
        out[body] = (ours, nasa)
        print(
            f"{body:5s}: n={N_SITES} sites x {N_TIMES} times | Pearson r {pearsonr(ours, nasa).statistic:.3f}  "
            f"Spearman {spearmanr(ours, nasa).statistic:.3f} | mean abs diff {np.abs(err).mean()*100:.1f} pts  "
            f"median {np.median(np.abs(err))*100:.1f} pts  bias {err.mean()*100:+.1f} pts  90th pct {np.percentile(np.abs(err), 90)*100:.1f} pts"
        )
    for name, a, b in (("Shackleton connecting ridge", -89.45, -137.0), ("Malapert Massif", -85.95, 2.5), ("de Gerlache rim", -88.7, -66.0)):
        print(f"  {name}: NASA sunlit {float(ref_sun.sample(np.array(a), np.array(b)))*100:.0f}%  Earth {float(ref_earth.sample(np.array(a), np.array(b)))*100:.0f}%")
    return out


if __name__ == "__main__":
    main()
