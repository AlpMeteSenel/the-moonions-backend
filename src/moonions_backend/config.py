from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOONIONS_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
TEXTURES = DATA / "textures"
MODELS = DATA / "models"
DB_PATH = DATA / "moonions.sqlite"

for _p in (CACHE, TEXTURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

MOON_RADIUS_M = 1_737_400.0
MOON_G = 1.62  # m/s^2

CORS_ORIGINS = os.environ.get("MOONIONS_CORS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
