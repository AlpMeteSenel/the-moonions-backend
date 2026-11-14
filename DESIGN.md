# Lunar Landing Site Navigator — System Design

Two repositories, one product:

| Repo | Role |
| --- | --- |
| `the-moonions` | Next.js dashboard: 3D Moon, candidates, landing dates, scores, backups, comparison, proposals |
| `the-moonions-backend` | Python (FastAPI): datasets, ephemeris, terrain analysis, scoring, backup-site planner, proposals store, ML |

```
┌──────────────────────────── the-moonions (Next.js) ────────────────────────────┐
│  Globe view (three.js / R3F, self-hosted NASA textures, real sun lighting)     │
│  Site view  (local 3D terrain mesh, primary + backup sites, divert rings)      │
│  Candidate rail · Date & time scrubber · Scorecard · Windows chart · Compare   │
│  Proposals (pending / accepted / rejected)                                     │
└───────────────▲──────────────────────────── REST/JSON + static textures ───────┘
                │
┌───────────────┴────────── the-moonions-backend (FastAPI) ──────────────────────┐
│ api.py  ─ evaluate · timeline · compare · terrain patch · sun · proposals      │
│ scoring.py      factor scores → overall 0-100 + verdict (Go / Caution / No-go) │
│ backups.py      nearby fallback sites (tiers by divert distance, fuel cost)    │
│ ephemeris.py    Sun & Earth position from site (JPL DE421 + lunar PA frame)    │
│ horizon.py      terrain horizon per azimuth (real shadowing at the poles)      │
│ terrain.py      slope, roughness, hazard fraction from LOLA DEMs               │
│ dem.py          best-available DEM: LOLA polar 80 m → LOLA global 474 m → 1.9 km│
│ craters.py      Robbins 2018 crater DB density/proximity                       │
│ store.py        SQLite proposals                                                │
│ ml/             auto-labelled DEM patches → CNN hazard model (phase 2)         │
└───────────────▲────────────────────────────────────────────────────────────────┘
                │  scripts/download_data.sh  →  data/raw  (precached, gitignored)
                │  scripts/build_textures.py →  data/textures (served at /static)
```

## Datasets (all public, downloaded once)

| Dataset | Source | Used for |
| --- | --- | --- |
| LROC WAC colour mosaic 8k (CGI Moon Kit) | NASA SVS 4720 | Globe colour texture |
| LOLA displacement `ldem_16.tif` | NASA SVS 4720 | Globe relief + normal map, fallback DEM |
| LOLA GDR `LDEM_64` (≈474 m/px) | PDS Geosciences | Global terrain analysis + horizons |
| LOLA polar `LDEM_80S/80N_80M` | PDS Geosciences | High-detail polar terrain (Artemis region) |
| JPL DE421 + `moon_pa_de421` + `moon_080317.tf` | JPL / NAIF | Sun & Earth position from any site/time |
| Robbins 2018 crater database (1.3 M craters) | USGS Astrogeology | Crater hazard density |

## Sun & Earth conditions (`conditions.py`, `GET /api/conditions`)

The core of the challenge: Sun and Earth relative to the local skyline at a site and date.
* positions: JPL DE421 + lunar PA frame (real orbits, libration), topocentric from the site;
* skyline: LOLA horizon at 0.5° azimuth, 150 km, lunar curvature, no atmosphere/refraction;
* finite disks: Sun ≈0.53°, Earth ≈1.9° from their actual distances; power ∝ visible fraction of
  the solar disk (slow polar sunsets along ridges);
* eclipses: Earth covering the Sun (validated on the 3 Mar 2026 lunar eclipse);
* direct-to-Earth: Earth's centre ≥ 1° (configurable antenna mask) above the skyline;
* rise/set to the minute (10-min grid + interpolation), countdowns to sunset/Earthset,
  longest darkness, battery state of charge over the stay.
The same physics drives the region analysis and AI maps.

## Scoring: plain-language factors

Each factor is 0-100, with a one-line explanation the UI shows. All are computed from data;
the UI never needs scientific jargon.

| UI label | What it measures | Data |
| --- | --- | --- |
| **Ground safety** | Slope & roughness in the landing zone; share of unsafe ground | DEM |
| **Sunlight** | Sun above the *terrain* horizon at touchdown, at a good angle for the cameras (≈5-25°), and how many hours of daylight follow | DE421 + horizon |
| **Earth contact** | Earth above the terrain horizon (direct radio link) and for how much of the stay | DE421 + horizon |
| **Crater risk** | Density of mapped craters around the site, distance to large rims | Robbins |
| **Backup options** | Quality and count of safe fallback sites within divert reach | backups.py |
| **Science value** | Proximity to polar cold traps / curated science regions | DEM + curated table |

`overall = weighted mean`, but any **hard fail** (e.g. no sun, slope > 15°) caps it
and forces a *No-go* verdict. Weights are mission profiles (`balanced`, `safety-first`,
`science-first`) selectable in the UI.

## Backup landing sites (core requirement)

If the lander can't touch down at the primary point late in descent (hazard spotted, dust,
navigation error), it needs **pre-vetted alternatives close by**. For every evaluation the
backend returns a **landing plan** = primary + ranked backups:

1. Sample candidate points on rings around the primary out to the divert radius
   (default 3 km, configurable).
2. Fast terrain pre-screen of every sample (slope, roughness, crater density).
3. Pick the best K (default 5) with a **diversity constraint**: minimum spacing between
   backups and spread over different bearings, so a single local problem can't knock out
   all options.
4. Full evaluation for each chosen backup (own horizon → own sunlight & Earth contact at the
   same landing time).
5. Tier by divert distance: **Near** (≤ 0.5 km, last-second hop), **Mid** (≤ 1.5 km),
   **Far** (≤ divert radius, early retarget), and estimate **divert cost** in extra
   seconds of hover fuel and Δv (bang-bang lateral translation under lunar gravity).
6. The primary's *Backup options* score = how good and how cheap its best backups are.

The 3D site view draws the backups as numbered markers with tier-coloured divert rings.

## Proposal workflow

`select site → select date → evaluate → compare / review → accept | reject` with notes.
Proposals are stored in SQLite with a frozen snapshot of the evaluation (scores, backups,
data versions) so a decision is auditable. Switching candidates never loses state: the
frontend keeps a candidate store; proposals reference candidates.

## API (v1)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Status + which datasets are loaded |
| GET | `/api/sites` | Curated candidate sites (Artemis III regions, historic landings) |
| GET | `/api/sun?time=` | Sub-solar & sub-Earth points (globe lighting) |
| POST | `/api/evaluate` | Full evaluation of one site at one time incl. backups |
| GET | `/api/timeline` | Score / sun / Earth vs time for a site → best landing windows |
| POST | `/api/compare` | Evaluate several sites at one time |
| GET | `/api/terrain` | Local heightfield + slope grid for the 3D site view |
| GET/POST/PATCH/DELETE | `/api/proposals` | Proposal CRUD and accept/reject |
| GET | `/static/textures/*` | Globe textures produced by `build_textures.py` |

## AI (v2): landing-success model

**Mission success = AI touchdown safety × physics checks (light at landing, power through the
stay, Earth link).** The checks are exact (JPL DE421 ephemeris, terrain horizons, battery model)
and are computed, not learned. The network learns the genuinely uncertain part: will the lander
touch down safely, given terrain finer than the planning maps show.

### Ground truth: landing simulator (`ml/simulate.py`)
Monte-Carlo (256 landers) on LOLA **20 m** south-polar terrain:
* navigation scatter σ = 60 m; hazard detection & avoidance retargets to the nearest clear
  footprint (3×3 safe cells) within 100 m, 90 % reliable;
* slopes measured against the **GRAIL GRGM1200B geoid** (degree 600, equipotential incl. rotation),
  i.e. the true local horizontal; deflection of the vertical at the south pole: median 0.04°,
  max 0.28°;
* sub-grid roughness term (metre-scale slopes the 20 m map can't see);
* **low-gravity dynamic tip-over**: sideways touchdown speed (σ 0.35 m/s/axis) converts to rotation
  about the downhill pad (angular-momentum conservation); tips if ½v²h²/(k²+L²) > g·L(1−cos α),
  g from GM/r² at the site. HLS-class lander: pad radius 5.5 m, CG 12 m; crew-ops slope limit 12°.

### Model (`ml/td_model.py`, `ml/train_td.py`)
* **Physics-informed residual**: logit P = logit(same simulator on the 80 m map) + f(x); starts
  exactly at physics, can only add information.
* Terrain seen through a CNN on the 80 m patch and/or frozen **NASA-IBM Lunar Foundation Model**
  embeddings (`ml/lfm_embed.py`); 13 tabular terrain/crater/gravity features.
* **Ensembles of 5** per variant → mean + uncertainty. Variants cnn / lfm / both compared; deployed
  variant chosen on the validation region; geographic test split (60–120°E) never used for choices.
* Model card with metrics vs physics-80 m and rule baselines, calibration, training curves:
  `GET /api/ai/model-card`.

### Regions, not points (`regions.py`)
NASA plans landings as regions (Artemis III: 15 × 15 km) and targets a landing ellipse inside them.
For a region and time: AI touchdown × physics checks on a 250 m grid → viable area; **landing zones**
= connected patches with success ≥ 80 % that contain the ±180 m (3σ) ellipse, ellipse placed at the
most interior point; Z1 = primary target, other zones = backups with divert distance / decide-by;
region timeline = viable share + best spot over time → region landing windows; planner ranks
regions × times by best-spot success × √(viable share) and names a backup region.

### History
* v0 (hazard_cnn): 3-class hazard from coarse terrain, 85 % balanced acc. vs 70 % for slope thresholds.
* v1 (success_net): learned light/power/comms too. 97 % of labels were 0 (dark / no comms), the
  metrics saturated and touchdown ranking on test (0.82) was *worse* than physics on the 80 m map
  (0.89). Replaced by the v2 hybrid design above.

## Coordinate conventions

* Latitude −90…90, longitude east-positive −180…180, selenographic (Moon ME / DE421).
* Globe mapping (three.js): `x = cos(lat)·cos(lon)`, `y = sin(lat)`, `z = −cos(lat)·sin(lon)`,
  which matches an equirectangular texture with lon −180 at its left edge.
