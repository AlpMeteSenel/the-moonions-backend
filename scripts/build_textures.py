"""Build self-hosted globe textures from the downloaded NASA datasets.

Outputs (data/textures, served at /static/textures):
  moon_color_{4k,8k}.jpg    LROC WAC colour mosaic
  moon_normal_4k.jpg        tangent-space normal map from LOLA (crisp terminator shading)
  moon_height_2k.png        8-bit relief for vertex displacement (+ meta in textures.json)
  hazard.png             RGBA global ground-steepness overlay (LOLA 474 m slope)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from moonions_backend.config import MOON_RADIUS_M, RAW, TEXTURES  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def color():
    src = RAW / "lroc_color_poles_8k.tif"
    img = tifffile.imread(src)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    im = Image.fromarray(img[..., :3])
    im.save(TEXTURES / "moon_color_8k.jpg", quality=88, optimize=True)
    im.resize((4096, 2048), Image.LANCZOS).save(TEXTURES / "moon_color_4k.jpg", quality=88, optimize=True)
    im.resize((1024, 512), Image.LANCZOS).save(TEXTURES / "moon_color_1k.jpg", quality=85)
    print("color ok", im.size)


def relief():
    h = tifffile.imread(RAW / "ldem_16.tif").astype(np.float32) * 1000.0  # km → m
    H, W = h.shape
    # normal map at native 5760x2880 then resized to 4k
    dx = 2 * np.pi * MOON_RADIUS_M / W  # metres per pixel east-west at equator
    lat = np.radians(90 - (np.arange(H) + 0.5) * 180 / H)[:, None]
    coslat = np.clip(np.cos(lat), 0.05, 1)
    dh_dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * dx * coslat)
    dh_dy = np.zeros_like(h)
    dh_dy[1:-1] = (h[:-2] - h[2:]) / (2 * dx)  # row 0 = north, so north-minus-south
    k = 3.0  # relief exaggeration for shading
    n = np.stack([-dh_dx * k, -dh_dy * k, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rgb = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((4096, 2048), Image.LANCZOS).save(TEXTURES / "moon_normal_4k.jpg", quality=92)
    lo, hi = float(h.min()), float(h.max())
    g = ((h - lo) / (hi - lo) * 255).astype(np.uint8)
    Image.fromarray(g).resize((2048, 1024), Image.LANCZOS).save(TEXTURES / "moon_height_2k.png", optimize=True)
    print("relief ok", lo, hi)
    return {"height_min_m": lo, "height_max_m": hi}


def hazard():
    """Slope overlay. Uses LOLA LDEM_64 (474 m) if present, else ldem_16."""
    img = RAW / "ldem_64.img"
    if img.exists():
        H, W = 11520, 23040
        h = np.memmap(img, dtype="<i2", mode="r", shape=(H, W))
        scale = 0.5
        # shift so the left edge is lon -180 like the other textures
        roll = W // 2
        src = "LDEM_64"
    else:
        h = tifffile.imread(RAW / "ldem_16.tif").astype(np.float32)
        H, W = h.shape
        scale, roll, src = 1000.0, 0, "ldem_16"
    f = 5 if src == "LDEM_64" else 2  # exact integer block size (LDEM_64 → 4608x2304)
    out_w, out_h = W // f, H // f
    fy = fx = f
    res = 2 * np.pi * MOON_RADIUS_M / W
    out = np.zeros((out_h, out_w), np.float32)
    chunk = f * 64
    for r0 in range(0, H, chunk):
        r1 = min(H, r0 + chunk)
        a0, a1 = max(0, r0 - 1), min(H, r1 + 1)
        blk = np.asarray(h[a0:a1], dtype=np.float32) * scale
        if roll:
            blk = np.roll(blk, roll, axis=1)
        lat = np.radians(90 - (np.arange(a0, a1) + 0.5) * 180 / H)[:, None]
        gx = (np.roll(blk, -1, 1) - np.roll(blk, 1, 1)) / (2 * res * np.clip(np.cos(lat), 0.02, 1))
        gy = np.gradient(blk, res, axis=0)
        s = np.degrees(np.arctan(np.hypot(gx, gy)))[r0 - a0 : r0 - a0 + (r1 - r0)]
        n_out = (r1 - r0) // f
        s = s[: n_out * f].reshape(n_out, f, out_w, f)
        # 90th percentile-ish: blend mean and max so isolated steep walls still show
        out[r0 // f : r0 // f + n_out] = 0.5 * s.mean(axis=(1, 3)) + 0.5 * s.max(axis=(1, 3))
    # colour ramp: transparent green-ish (safe) → amber → red (unsafe). The colour depends on the
    # slope only, so store it as a 256-entry palette PNG (per-entry alpha), visually identical:
    # same look, a fraction of the size of full RGBA (which was ~30 MB).
    top = 20.0
    levels = np.linspace(0, top, 256, dtype=np.float32)  # full 8-bit palette: ~0.08° per step
    step = float(levels[1] - levels[0])
    t = np.clip(levels / top, 0, 1)
    pal = np.zeros((len(levels), 4), np.uint8)
    pal[:, 0] = (255 * np.clip(t * 2, 0, 1)).astype(np.uint8)
    pal[:, 1] = (255 * np.clip(2 - t * 2, 0, 1) * 0.85).astype(np.uint8)
    pal[:, 2] = (80 * (1 - t)).astype(np.uint8)
    pal[:, 3] = (40 + 200 * np.clip((levels - 3) / 15, 0, 1)).astype(np.uint8)
    idx = np.clip(np.rint(out / step), 0, len(levels) - 1).astype(np.uint8)
    img = Image.fromarray(idx, "P")
    img.putpalette(pal[:, :3].ravel().tolist())
    img.info["transparency"] = bytes(pal[:, 3].tolist())
    img.save(TEXTURES / "hazard.png", optimize=True, transparency=bytes(pal[:, 3].tolist()))
    print("hazard ok from", src)
    return {"hazard_source": src, "hazard_max_slope_deg": 20.0}


def caps():
    """Polar caps (|lat| >= 80) in polar-stereographic projection from LOLA 80 m DEMs.

    The equirectangular globe texture pinches at the poles, exactly where most Artemis
    candidates are, so the dashboard draws these caps on top.
    """
    from moonions_backend.dem import PolarImg

    meta = {}
    color = np.asarray(Image.open(TEXTURES / "moon_color_8k.jpg"), dtype=np.float32)
    CH, CW = color.shape[:2]
    for pole in ("s", "n"):
        img, lbl = RAW / f"ldem_80{pole}_80m.img", RAW / f"ldem_80{pole}_80m.lbl"
        if not img.exists():
            continue
        src = PolarImg(img, lbl)
        f = 2  # 160 m/px working resolution
        h = np.asarray(src.arr, dtype=np.float32)[: (src.lines // f) * f, : (src.samples // f) * f]
        h = h.reshape(src.lines // f, f, src.samples // f, f).mean(axis=(1, 3)) * src.scale
        res = src.map_scale * f
        n_px = h.shape[0]
        gx = np.gradient(h, res, axis=1)  # +col = +x
        gy = -np.gradient(h, res, axis=0)  # +row = -y
        k = 3.0
        nrm = np.stack([-gx * k, -gy * k, np.ones_like(h)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8)).save(TEXTURES / f"cap_{pole}_normal.jpg", quality=92)
        lo, hi = float(h.min()), float(h.max())
        hh = Image.fromarray(((h - lo) / (hi - lo) * 255).astype(np.uint8)).resize((1024, 1024), Image.LANCZOS)
        hh.save(TEXTURES / f"cap_{pole}_height.png", optimize=True)
        # albedo: resample LROC colour into the stereographic grid
        m = 2048
        half = src.sample_off * src.map_scale  # metres from centre to edge
        ax = (np.arange(m) + 0.5) / m * 2 * half - half
        X, Y = np.meshgrid(ax, -ax)  # row 0 = +y
        rho = np.hypot(X, Y)
        R = src.min_abs_lat  # unused, keeps linters calm
        del R
        from moonions_backend.config import MOON_RADIUS_M as RM

        colat = 2 * np.arctan(rho / (2 * RM))
        if src.south:
            lat = -(90 - np.degrees(colat))
            lon = np.degrees(np.arctan2(X, Y))
        else:
            lat = 90 - np.degrees(colat)
            lon = np.degrees(np.arctan2(X, -Y))
        cx = ((lon + 180) / 360 * CW - 0.5) % CW
        cy = np.clip((90 - lat) / 180 * CH - 0.5, 0, CH - 1)
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        fx, fy = (cx - x0)[..., None], (cy - y0)[..., None]
        x1, y1 = (x0 + 1) % CW, np.minimum(y0 + 1, CH - 1)
        c = (color[y0, x0] * (1 - fx) + color[y0, x1] * fx) * (1 - fy) + (color[y1, x0] * (1 - fx) + color[y1, x1] * fx) * fy
        # The equirectangular source pinches into radial streaks near the pole: fade to the
        # regional mean tone poleward of 86° and let the LOLA normal map carry the detail.
        from scipy.ndimage import gaussian_filter

        blurred = np.stack([gaussian_filter(c[..., k], 10) for k in range(3)], -1)
        wb = np.clip((np.abs(lat) - 80.6) / 1.5, 0, 1)[..., None]
        c = c * (1 - wb) + blurred * wb  # removes meridian streaks, keeps broad albedo
        ring = (np.abs(lat) > 84) & (np.abs(lat) < 86)
        mean_tone = c[ring].mean(axis=0) if ring.any() else c.reshape(-1, 3).mean(0)
        w = np.clip((np.abs(lat) - 86.0) / 2.0, 0, 1)[..., None]
        w = w * w * (3 - 2 * w)
        c = c * (1 - w) + mean_tone * w
        Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(TEXTURES / f"cap_{pole}_color.jpg", quality=88)
        meta[f"cap_{pole}"] = {"height_min_m": lo, "height_max_m": hi, "half_size_m": half, "min_abs_lat": src.min_abs_lat}
        print("cap", pole, "ok", n_px, lo, hi)
    return {"caps": meta}


if __name__ == "__main__":
    meta = {}
    what = set(sys.argv[1:]) or {"color", "relief", "hazard", "caps"}
    if "color" in what:
        color()
    if "relief" in what:
        meta.update(relief())
    if "hazard" in what:
        meta.update(hazard())
    if "caps" in what:
        meta.update(caps())
    mpath = TEXTURES / "textures.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    old.update(meta)
    mpath.write_text(json.dumps(old, indent=2))
