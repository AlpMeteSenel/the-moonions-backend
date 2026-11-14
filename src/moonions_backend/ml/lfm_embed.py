"""DEM-derived Lunar Foundation Model embeddings (no fine tuning).

Inputs are north-up BCHW, DTM metres relative to 1737.4 km and slope degrees.
Aspect channels are [sin, cos]; the installed source does not define the angle
origin, so we assume downslope azimuth clockwise from north (east, north).
Normalization is exactly config.data.domains.* scaler=std, without clipping or
per-tile elevation centering. Native WAC-family tiles are 853*60 m, resized to
512 then cropped to 256: about 100 m/pixel, not a full-tile resize to 256.
We sample directly at 100 m, so DEM filtering differs from upstream imagery.

The TerraTorch wrapper returns 12 (B,N,D) layers; the last is encoder-normalized.
There is no CLS token; config has zero registers (we request their removal).
We average the three modality streams at matching row-major patch positions
AFTER joint attention, retaining 768 features. Center uses the central 2x2
patches at ps16: 3.2 km across at 100 m/px, the nearest symmetric patch-aligned
support to the requested ~2.4 km. Attention still includes the entire tile.

Examples::

    uv run python -m moonions_backend.ml.lfm_embed --random 400 --sanity
    uv run python -m moonions_backend.ml.lfm_embed --from-dataset
    uv run python -m moonions_backend.ml.lfm_embed --sites sites.npz

NPZ sites accept lat/lon arrays or meta[:, :2]; CSV requires lat,lon headers.
Sanity CV uses a fixed ridge penalty and train-fold-only feature scaling;
random spatial folds measure interpolation, not geographic generalization.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
import time

import numpy as np

from moonions_backend.dem import get_dem
from moonions_backend.terrain import analyze_zone, slope_grid

_ROOT = Path(__file__).resolve().parents[3]
_MODEL = _ROOT / "data/models/lfm/backbone"
_MODALITIES = ("dtm", "slope", "aspect")


@lru_cache(maxsize=1)
def _domains():
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(_MODEL / "config.yaml").data.domains)


@lru_cache(maxsize=4)
def load_encoder(device=None, patch_size=16):
    """Build/cache an eval backbone; default to CUDA, otherwise CPU.

    patch_size is the wrapper's FlexiViT constructor override (checkpoint
    projection weights are PI-resized by the installed embedding loader).
    """
    import torch
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
    import terratorch_integration  # noqa: F401: registers ni_lfm_v1_base

    if not isinstance(patch_size, int) or patch_size <= 0 or 256 % patch_size:
        raise ValueError("patch_size must be a positive divisor of 256")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = TERRATORCH_BACKBONE_REGISTRY.build(
        "ni_lfm_v1_base", modalities=list(_MODALITIES),
        cfg=str(_MODEL / "config.yaml"), checkpoint_path=str(_MODEL / "checkpoint.pt"),
        patch_size=patch_size, merge_method="mean", remove_register_tokens=True,
    )
    return encoder.requires_grad_(False).eval().to(device)


def _coordinates(lats, lons):
    lat, lon = np.atleast_1d(np.asarray(lats, dtype=np.float64)), np.atleast_1d(np.asarray(lons, dtype=np.float64))
    if lat.ndim != 1 or lon.shape != lat.shape:
        raise ValueError("lats and lons must be matching one-dimensional arrays")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or np.any(np.abs(lat) > 90):
        raise ValueError("coordinates must be finite, with latitude in [-90, 90]")
    return lat, lon


def make_tiles(lats, lons, size=256, res_m=100.0):
    """Return normalized float32 arrays (N,C,size,size), site between center pixels."""
    lat, lon = _coordinates(lats, lons)
    if not isinstance(size, int) or size < 2 or not np.isfinite(res_m) or res_m <= 0:
        raise ValueError("size must be >=2 and res_m must be positive")
    result = {m: np.empty((len(lat), 2 if m == "aspect" else 1, size, size), np.float32) for m in _MODALITIES}
    for i, (la, lo) in enumerate(zip(lat, lon)):
        _, _, h, _ = get_dem().local_grid(float(la), float(lo), (size - 1) * res_m / 2, res_m)
        if not np.isfinite(h).all():
            raise ValueError(f"Nonfinite DEM at site {i}: {la}, {lo}")
        gy, gx = np.gradient(h.astype(np.float64), res_m)
        # Rows run southward: downslope east=-gx, north=+gy.
        angle = np.arctan2(-gx, gy)
        raw = {"dtm": h[None], "slope": slope_grid(h, res_m)[None],
               "aspect": np.stack((np.sin(angle), np.cos(angle)))}
        for m in _MODALITIES:
            info = _domains()[m]
            if info["scaler"] != "std":
                raise ValueError(f"Unexpected {m} scaler: {info['scaler']}")
            mean = np.asarray(info["stats"]["mean"], np.float32)[:, None, None]
            std = np.asarray(info["stats"]["std"], np.float32)[:, None, None]
            result[m][i] = (raw[m].astype(np.float32) - mean) / std
    return result


def embed(lats, lons, batch=8):
    """Return center/global float16 (N,768) arrays, streaming small batches.

    Internally cap CUDA batches at two for the 4 GB GPU; batch is an upper
    bound. CPU fallback is supported but substantially slower.
    """
    import torch
    lat, lon = _coordinates(lats, lons)
    if not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    result = {k: np.empty((len(lat), 768), np.float16) for k in ("center", "global")}
    if not len(lat):
        return result
    encoder = load_encoder()
    device = next(encoder.parameters()).device
    step = min(batch, 8)
    for start in range(0, len(lat), step):
        tiles = make_tiles(lat[start:start + step], lon[start:start + step])
        inputs = {m: torch.from_numpy(v).to(device) for m, v in tiles.items()}
        amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with torch.no_grad(), amp:
            tokens = encoder(inputs)[-1]
            if tokens.shape[1:] != (256, 768):
                raise RuntimeError(f"Unexpected merged token layout: {tokens.shape}")
            grid = tokens.reshape(-1, 16, 16, 768)
            pooled = {"center": grid[:, 7:9, 7:9].float().mean((1, 2)),
                      "global": tokens.float().mean(1)}
            for k, v in pooled.items():
                if not torch.isfinite(v).all():
                    raise RuntimeError("Nonfinite embeddings")
                result[k][start:start + len(v)] = v.cpu().numpy().astype(np.float16)
    return result


def _ridge_cv(x, y, seed=0, alpha=100.0):
    """Closed-form dual ridge, five held-out folds; return pooled OOF R²."""
    if len(x) < 5:
        raise ValueError("Sanity check requires at least five sites")
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    pred = np.empty_like(y)
    for test in np.array_split(np.random.default_rng(seed).permutation(len(x)), 5):
        train = np.setdiff1d(np.arange(len(x)), test)
        mean, std = x[train].mean(0), x[train].std(0)
        std[std < 1e-8] = 1
        a, b = (x[train] - mean) / std, (x[test] - mean) / std
        ym = y[train].mean(0)
        weights = np.linalg.solve(a @ a.T + alpha * np.eye(len(train)), y[train] - ym)
        pred[test] = b @ a.T @ weights + ym
    denominator = ((y - y.mean(0)) ** 2).sum(0)
    return np.where(denominator > 0, 1 - ((y - pred) ** 2).sum(0) / denominator, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--sites", help="NPZ/CSV path, or RANDOM (400 sites)")
    sources.add_argument("--random", type=int, metavar="N")
    sources.add_argument("--from-dataset", action="store_true")
    parser.add_argument("--out", type=Path, default=_ROOT / "data/cache/lfm_embeddings.npz")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sanity", action="store_true", help="Print five-fold ridge R² for selected sites (use --random 400)")
    args = parser.parse_args()
    if args.random is not None or (args.sites and args.sites.upper() == "RANDOM"):
        n = args.random if args.random is not None else 400
        if n <= 0:
            parser.error("--random must be positive")
        rng = np.random.default_rng(args.seed)
        # Equal area on a sphere: sin(latitude), not latitude, is uniform.
        lat = np.degrees(np.arcsin(rng.uniform(-1, np.sin(np.radians(-80.6)), n)))
        lon = rng.uniform(-180, 180, n)
    else:
        path = _ROOT / "data/cache/success_dataset.npz" if args.from_dataset else Path(args.sites)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                coords = data["meta"][:, :2] if "meta" in data else np.column_stack((data["lat"], data["lon"]))
        else:
            data = np.genfromtxt(path, delimiter=",", names=True)
            coords = np.column_stack((np.atleast_1d(data["lat"]), np.atleast_1d(data["lon"])))
        lat, lon = np.unique(coords, axis=0).T
    import torch
    encoder = load_encoder()
    device = next(encoder.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    embeddings = embed(lat, lon, args.batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, lat=lat, lon=lon, **embeddings)
    peak = f"{torch.cuda.max_memory_allocated(device) / 2**30:.3f} GiB allocated" if device.type == "cuda" else "N/A (CPU)"
    print(f"Saved {len(lat)} sites to {args.out}; {len(lat)/elapsed:.2f} sites/s (DEM + inference, excludes model load); peak VRAM {peak}", flush=True)
    if args.sanity:
        targets = []
        for la, lo in zip(lat, lon):
            zone = analyze_zone(float(la), float(lo), 300)
            if not np.isclose(zone.resolution_m, 80):
                raise ValueError("Sanity target requires the 80 m polar DEM")
            tile = make_tiles([la], [lo])["slope"]
            stats = _domains()["slope"]["stats"]
            targets.append((zone.slope_p90_deg, float(tile.mean() * stats["std"][0] + stats["mean"][0])))
        scores = _ridge_cv(embeddings["center"], targets, args.seed)
        print(f"5-fold CV R² (ridge alpha=100): 300 m zone slope_p90={scores[0]:.4f}; mean tile slope={scores[1]:.4f}")
    print("Caveats: inferred aspect origin (downslope clockwise from north); central 2x2 support=3.2 km; direct DEM sampling differs from pretraining resize/filtering.")


if __name__ == "__main__":
    main()
