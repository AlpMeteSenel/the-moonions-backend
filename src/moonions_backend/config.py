from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOONIONS_ROOT", Path(__file__).resolve().parents[2]))


def _load_dotenv(path: Path) -> None:
    """Read KEY=value lines from the backend's .env (local secrets) without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")
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

# "*" by default so the dashboard works when either side is tunnelled (e.g. ngrok); restrict via MOONIONS_CORS
CORS_ORIGINS = os.environ.get("MOONIONS_CORS", "*").split(",")
