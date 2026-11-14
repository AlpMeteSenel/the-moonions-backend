#!/usr/bin/env bash
# Full AI pipeline: validate truth DEM → simulate labels → foundation-model embeddings → train & compare.
set -euo pipefail
cd "$(dirname "$0")/.."
SITES=${SITES:-8000}
echo "== 1. validate LOLA 20 m projection against the 80 m map"
uv run python - <<'PY'
import numpy as np
from moonions_backend.dem import get_dem
from moonions_backend.ml.simulate import get_truth_dem
t = get_truth_dem(); p = [s for s in get_dem().polar if s.south][0]
rng = np.random.default_rng(1); lat = -rng.uniform(80.6, 89.9, 3000); lon = rng.uniform(-180, 180, 3000)
a, b = p.sample(lat, lon), t.sample(lat, lon)
r = np.corrcoef(a, b)[0, 1]; print(f"corr {r:.4f}  mean |diff| {np.mean(np.abs(a-b)):.1f} m")
assert r > 0.99, "20 m DEM projection mismatch"
PY
echo "== 2. simulated landing dataset ($SITES sites x 4 dates)"
uv run python -m moonions_backend.ml.success_dataset --sites "$SITES"
echo "== 3. NASA-IBM Lunar Foundation Model embeddings"
uv run python -m moonions_backend.ml.lfm_embed --from-dataset
echo "== 4. train + compare variants"
uv run python -m moonions_backend.ml.train_success --variants cnn,lfm,both --epochs 40
