"""Landing-mission simulator: the ground truth the AI learns from.

For a (site, touchdown time) it estimates the probability the mission succeeds, using the most
detailed NASA data available (LOLA 20 m south-polar DEM) and a Monte-Carlo landing model:

* touchdown: N landers scattered by navigation error (Gaussian, σ per axis). Each lands on its
  touchdown cell if that cell is safe (slope at 20 m baseline below the lander limit); otherwise
  its onboard hazard detection can shift it to the nearest safe cell within HDA_HOP_M.
* light at landing: Sun above the terrain skyline (horizon from the DEM).
* energy: battery state-of-charge simulated over the surface stay (solar panels charge while lit,
  constant load) — fails if the battery empties.
* comms: Earth above the skyline for at least COMMS_MIN_FRAC of the stay (direct-to-Earth link).

success = P(touchdown) × lit × energy_ok × comms_ok

v2: slopes against the GRAIL geoid, hazard-avoidance retargeting, sub-grid roughness, gear limit and
low-gravity dynamic tip-over (see `tipover`).

This is deliberately *more detailed* than the planning features the models see (80 m terrain),
so the AI has to learn sub-resolution behaviour, and it replaces an expensive simulation with a
fast surrogate that can scan the whole polar region for any date.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..config import RAW
from ..dem import PolarImg, get_dem
from ..terrain import slope_grid


@dataclass(frozen=True)
class SimParams:
    # guidance & hazard avoidance
    nav_sigma_m: float = 60.0  # 1σ per axis touchdown dispersion (≈180 m 3σ, Artemis-class precision landing)
    hda_hop_m: float = 100.0  # onboard hazard detection & avoidance retarget radius
    hda_safe_slope_deg: float = 8.0  # HDA treats 20 m cells steeper than this as hazards
    hda_reliability: float = 0.9  # chance a needed divert is detected and flown successfully
    # lander (HLS-class crewed lander; change for CLPS-class: pad 2.2 m, CG 2.0 m)
    footpad_radius_m: float = 5.5  # distance from the vertical axis to a footpad
    cg_height_m: float = 12.0  # centre of gravity above the footpads
    gyration_ratio: float = 0.45  # radius of gyration about the CG as a fraction of pad→CG distance
    gear_max_slope_deg: float = 12.0  # crew-ops limit: levelling, egress and ascent from a tilted lander
    lateral_sigma_mps: float = 0.35  # residual horizontal touchdown speed, 1σ per axis
    subgrid_slope_deg: float = 1.5  # extra slope from metre-scale roughness the 20 m map can't see
    n_mc: int = 256
    # surface stay
    battery_kwh: float = 100.0
    panel_kw: float = 5.0  # vertical panels: output ≈ panel_kw·cos(sun el) when lit
    load_kw: float = 2.0
    comms_min_frac: float = 0.5
    comm_mask_deg: float = 1.0  # Earth's centre must clear the skyline by this much for direct-to-Earth
    mission_hours: float = 156.0
    step_hours: float = 2.0

    def to_dict(self):
        return asdict(self)


DEFAULT = SimParams()


@lru_cache(maxsize=1)
def get_truth_dem() -> PolarImg | None:
    img, lbl = RAW / "ldem_80s_20m.img", RAW / "ldem_80s_20m.lbl"
    if not (img.exists() and lbl.exists()):
        return None
    return PolarImg(img, lbl)


@lru_cache(maxsize=1)
def _mc(n: int, seed: int = 7):
    """Fixed Monte-Carlo draws (common random numbers → smooth, comparable labels)."""
    r = np.random.default_rng(seed)
    return {"pos": r.normal(0, 1, (n, 2)), "vel": r.normal(0, 1, (n, 2)), "sub": r.normal(0, 1, n), "hda": r.uniform(0, 1, n)}


def tipover(slope_deg: np.ndarray, v_downhill: np.ndarray, g: float, p: SimParams = DEFAULT) -> np.ndarray:
    """True where the lander tips over.

    Rigid lander pivoting about its downhill (or uphill) footpad. On a slope θ the CG sits at angle
    α = φs − θ from the vertical above the downhill pad (φs = atan(pad/CG), the static tip angle).
    When the pad catches, angular momentum about the pad is conserved: m·v·h = I_p·ω with
    h = L·cos α the CG height above the pad and I_p = m(k² + L²). The lander tips if that rotational
    energy can lift the CG over the pad: ½·v²·h²/(k² + L²) > g·L·(1 − cos α).
    Lower lunar gravity makes the same sideways speed ~6x more dangerous than on Earth.
    """
    a, H = p.footpad_radius_m, p.cg_height_m
    L = np.hypot(a, H)
    k2 = (p.gyration_ratio * L) ** 2
    phi = np.arctan2(a, H)
    th = np.radians(slope_deg)
    downhill = v_downhill >= 0
    alpha = np.where(downhill, phi - th, phi + th)  # moving uphill tips over the uphill pad
    static = alpha <= 0
    h = L * np.cos(np.clip(alpha, 0, None))
    ke = 0.5 * v_downhill**2 * h**2 / (k2 + L**2)
    need = g * L * (1 - np.cos(np.clip(alpha, 0, None)))
    return static | (ke > need)


def touchdown(lat: float, lon: float, p: SimParams = DEFAULT, truth: PolarImg | None = None) -> dict:
    """Monte-Carlo touchdown on the given DEM (default: LOLA 20 m truth).

    Slopes are measured against the GRAIL geoid (equipotential), not a sphere.
    """
    from ..dem import offset_latlon
    from ..gravity import geoid_height, surface_gravity

    truth = truth or get_truth_dem()
    if truth is None:
        raise FileNotFoundError("ldem_80s_20m not downloaded")
    dem = get_dem()
    res = getattr(truth, "map_scale", None) or truth.resolution_m  # polar: 20/80 m, global: 474 m
    half = 4 * p.nav_sigma_m + p.hda_hop_m + 3 * res
    east, north, h, _ = dem.local_grid(lat, lon, half, res, source=truth)
    E, Nn = np.meshgrid(east, north)
    glat, glon = offset_latlon(lat, lon, E, Nn)
    h_eq = h - geoid_height(glat, glon)  # height above the equipotential surface
    gy, gx = np.gradient(h_eq.astype(np.float64), res)  # rows run north→south
    g_north, g_east = -gy, gx
    grad = np.hypot(g_east, g_north)
    slope = np.degrees(np.arctan(grad))
    # metre-scale roughness proxy: local variability of the 20 m slope
    from scipy.ndimage import uniform_filter

    m1 = uniform_filter(slope, 3)
    m2 = uniform_filter(slope**2, 3)
    rough = np.sqrt(np.clip(m2 - m1**2, 0, None))
    hazard = slope > p.hda_safe_slope_deg
    # a valid divert target needs a clear footprint: the cell and its 8 neighbours must be safe
    from scipy.ndimage import binary_dilation

    blocked = binary_dilation(hazard, np.ones((3, 3), bool))
    dist, (ri, ci) = distance_transform_edt(blocked, return_indices=True)
    dist = dist * res

    mc = _mc(p.n_mc)
    n = h.shape[0]
    off = mc["pos"] * p.nav_sigma_m
    cols = np.clip(np.round((off[:, 0] + half) / res).astype(int), 0, n - 1)
    rows = np.clip(np.round((half - off[:, 1]) / res).astype(int), 0, n - 1)
    # hazard avoidance: if the cell under the lander is a hazard and a safe one is within reach, divert
    divert = blocked[rows, cols] & (dist[rows, cols] <= p.hda_hop_m) & (mc["hda"] < p.hda_reliability)
    fr = np.where(divert, ri[rows, cols], rows)
    fc = np.where(divert, ci[rows, cols], cols)
    sl = slope[fr, fc]
    eff = np.clip(sl + (p.subgrid_slope_deg + 0.5 * rough[fr, fc]) * mc["sub"], 0, None)
    # downhill unit vector at the final cell, and the lander's sideways speed along it
    ge, gn = g_east[fr, fc], g_north[fr, fc]
    norm = np.hypot(ge, gn) + 1e-12
    v = mc["vel"] * p.lateral_sigma_mps
    v_down = -(v[:, 0] * ge + v[:, 1] * gn) / norm
    g = float(surface_gravity(lat, float(h[n // 2, n // 2])))
    tips = tipover(eff, v_down, g, p)
    gear = eff > p.gear_max_slope_deg
    ok = ~tips & ~gear
    c = n // 2
    return {
        "p_touchdown": float(ok.mean()),
        "p_direct": float((~hazard[rows, cols]).mean()),
        "p_tipover": float(tips.mean()),
        "p_gear": float(gear.mean()),
        "p_diverted": float(divert.mean()),
        "truth_slope_center": float(slope[c, c]),
        "truth_unsafe_frac": float(hazard.mean()),
        "g_local": g,
    }


def energy_soc(sun_el: np.ndarray, lit: np.ndarray, p: SimParams = DEFAULT) -> np.ndarray:
    """Battery state of charge (0..1) through the stay, starting full. sun_el, lit: (N, T)."""
    gen = p.panel_kw * np.cos(np.radians(np.clip(sun_el, 0, 90))) * lit
    net_kwh = (gen - p.load_kw) * p.step_hours
    soc = np.empty_like(net_kwh)
    cur = np.full(net_kwh.shape[0], p.battery_kwh)
    for t in range(net_kwh.shape[1]):
        cur = np.clip(cur + net_kwh[:, t], 0, p.battery_kwh)
        soc[:, t] = cur
    return soc / p.battery_kwh


def time_outcomes(sun_clear: np.ndarray, sun_el: np.ndarray, earth_clear: np.ndarray, p: SimParams = DEFAULT, sun_frac: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Deterministic lighting / energy / comms outcomes for (N, T) clearance series (T = stay steps).

    `sun_frac` (visible fraction of the solar disk incl. eclipses, see conditions.py) makes
    sunlight and power physical; without it the Sun's centre clearance is used.
    """
    if sun_frac is None:
        sun_frac = (sun_clear > 0.5).astype(float)
    lit = sun_frac > 0.5
    vis = earth_clear >= p.comm_mask_deg
    soc = energy_soc(sun_el, sun_frac, p)
    dark = ~lit
    # longest continuous dark spell (hours)
    longest = np.zeros(lit.shape[0])
    run = np.zeros(lit.shape[0])
    for t in range(lit.shape[1]):
        run = np.where(dark[:, t], run + p.step_hours, 0)
        longest = np.maximum(longest, run)
    return {
        "lit0": lit[:, 0].astype(float),
        "lit_frac": lit.mean(1),
        "longest_dark_h": longest,
        "min_soc": soc.min(1),
        "energy_ok": (soc.min(1) > 0.0).astype(float),
        "earth_frac": vis.mean(1),
        "earth0": vis[:, 0].astype(float),
        "comms_ok": (vis.mean(1) >= p.comms_min_frac).astype(float),
    }
