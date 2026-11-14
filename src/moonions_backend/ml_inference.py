"""Optional CNN hazard model inference.

If a trained checkpoint (data/models/hazard_cnn.pt) and PyTorch are present, scoring blends
the model's safe-landing probability into Ground safety. Otherwise this returns None and the
rule-based scoring is used alone (phase 1).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .config import MODELS

log = logging.getLogger(__name__)
CKPT = MODELS / "hazard_cnn.pt"


@lru_cache(maxsize=1)
def _load():
    if not CKPT.exists():
        return None
    try:
        import torch

        from .ml.model import HazardCNN

        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        m = HazardCNN(in_ch=ck["in_ch"], n_classes=ck["n_classes"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return m, ck
    except Exception as exc:  # torch not installed, incompatible checkpoint, ...
        log.warning("hazard model unavailable: %s", exc)
        return None


def available() -> bool:
    return _load() is not None


def predict(lat: float, lon: float) -> dict | None:
    loaded = _load()
    if loaded is None:
        return None
    import torch

    from .ml.dataset import patch_features

    model, ck = loaded
    x = patch_features(lat, lon, ck["patch"], ck["patch_res_m"])
    if x is None:
        return None
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(x[None]).float()), dim=1)[0].numpy()
    classes = ck["classes"]
    return {"p_safe": float(p[classes.index("safe")]), "probs": {c: round(float(v), 3) for c, v in zip(classes, p)}, "model": ck.get("version", "hazard_cnn")}
