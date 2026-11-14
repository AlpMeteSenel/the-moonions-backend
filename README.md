# the-moonions-backend

Python (FastAPI) backend for the **Lunar Landing Site Navigator**: real lunar datasets, sun/Earth
ephemeris, terrain analysis, landing suitability scoring, nearby **backup landing sites**,
landing proposals, and the hazard CNN. Frontend: [`the-moonions`](https://github.com/AlpMeteSenel/the-moonions).

See [DESIGN.md](DESIGN.md) for the architecture, datasets and scoring model.

## Setup

```bash
uv sync                               # core deps (add --extra ml for PyTorch)
./scripts/download_data.sh            # ~1.2 GB of NASA/JPL/USGS data into data/raw (re-runnable)
uv run python scripts/build_textures.py   # globe + polar-cap + hazard textures into data/textures
uv run uvicorn moonions_backend.api:app --port 8000 --reload --reload-dir src
```

Open http://localhost:8000/docs for the interactive API.

## Phase 2: hazard CNN

```bash
uv sync --extra ml
uv run python -m moonions_backend.ml.dataset --n 20000   # auto-labelled patches (~2 min)
uv run python -m moonions_backend.ml.train --epochs 30   # writes data/models/hazard_cnn.pt
```

When `data/models/hazard_cnn.pt` exists the API blends the model into *Ground safety* automatically.

## Data used

| Dataset | Use |
| --- | --- |
| LOLA LDEM_64 (474 m) + polar LDEM_80S/80N (80 m) | terrain, slopes, horizons, backups |
| NASA SVS CGI Moon Kit (LROC colour, LOLA displacement) | 3D globe |
| JPL DE421 + lunar PA frame | Sun & Earth positions from any site and time |
| Robbins 2018 crater DB (optional, USGS) | crater risk; neutral score if missing |
