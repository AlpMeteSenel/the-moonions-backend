"""FastAPI application."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ai, ai_polar, ml_inference, regions, store
from .backups import suggest_in_region
from .config import CORS_ORIGINS, TEXTURES
from .craters import get_craters
from .dem import get_dem
from .ephemeris import sun_and_earth_points
from .scoring import FACTOR_LABELS, PROFILES, EvalParams, evaluate_site, timeline
from .sites import SITES
from .terrain import terrain_patch

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Lunar Landing Site Navigator", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.mount("/static/textures", StaticFiles(directory=TEXTURES), name="textures")

Profile = Literal["balanced", "safety-first", "science-first"]


class EvalRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=360)
    landing_time: datetime
    mission_hours: float = Field(156.0, ge=1, le=24 * 30)
    zone_radius_m: float = Field(300.0, ge=50, le=5000)
    divert_radius_m: float = Field(3000.0, ge=300, le=10000)
    n_backups: int = Field(5, ge=0, le=10)
    profile: Profile = "balanced"

    def params(self) -> EvalParams:
        return EvalParams(
            landing_time=self.landing_time,
            mission_hours=self.mission_hours,
            zone_radius_m=self.zone_radius_m,
            divert_radius_m=self.divert_radius_m,
            n_backups=self.n_backups,
            profile=self.profile,
        )


class Point(BaseModel):
    id: str | None = None
    name: str | None = None
    lat: float
    lon: float
    half_size_m: float | None = Field(None, ge=500, le=20000)


class CompareRequest(BaseModel):
    sites: list[Point] = Field(min_length=1, max_length=8)
    landing_time: datetime
    mission_hours: float = 156.0
    divert_radius_m: float = 3000.0
    profile: Profile = "balanced"


class SuggestRequest(EvalRequest):
    region_radius_m: float = Field(7500.0, ge=1000, le=30000)
    n: int = Field(4, ge=1, le=8)


class ProposalCreate(BaseModel):
    name: str
    evaluation: dict
    notes: str = ""


class ProposalUpdate(BaseModel):
    status: Literal["pending", "accepted", "rejected"] | None = None
    notes: str | None = None
    decided_by: str | None = None
    name: str | None = None


def _norm_lon(lon: float) -> float:
    return round((lon + 180.0) % 360.0 - 180.0, 8)


@app.get("/api/health")
def health():
    dem = get_dem()
    return {
        "ok": True,
        "dem_sources": [{"name": s.name, "resolution_m": round(s.resolution_m, 1)} for s in dem.sources],
        "craters": get_craters() is not None,
        "ai_model": ml_inference.available(),
        "textures": sorted(p.name for p in TEXTURES.glob("*") if p.is_file()),
        "profiles": PROFILES,
        "factor_labels": FACTOR_LABELS,
    }


@app.get("/api/sites")
def sites():
    return SITES


@app.get("/api/sun")
def sun(time: datetime | None = None):
    t = time or datetime.now(timezone.utc)
    return {"time": t.isoformat(), **sun_and_earth_points(t)}


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    return evaluate_site(req.lat, _norm_lon(req.lon), req.params())


@app.post("/api/suggest")
def suggest(req: SuggestRequest):
    return suggest_in_region(req.lat, _norm_lon(req.lon), req.region_radius_m, req.params(), n=req.n)


@app.post("/api/compare")
def compare(req: CompareRequest):
    p = EvalParams(landing_time=req.landing_time, mission_hours=req.mission_hours, divert_radius_m=req.divert_radius_m, profile=req.profile)
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda s: evaluate_site(s.lat, _norm_lon(s.lon), p), req.sites))
    for s, r in zip(req.sites, results):
        r["id"], r["name"] = s.id, s.name
    return results


@app.get("/api/timeline")
def get_timeline(
    lat: float,
    lon: float,
    start: datetime,
    days: float = Query(30.0, gt=0, le=120),
    step_hours: float = Query(2.0, ge=0.5, le=24),
    mission_hours: float = Query(156.0, ge=1, le=720),
):
    return timeline(lat, _norm_lon(lon), start, days, step_hours, mission_hours)


@app.get("/api/terrain")
def get_terrain(lat: float, lon: float, half_size_m: float = Query(4000.0, ge=500, le=30000), cells: int = Query(161, ge=33, le=401)):
    return terrain_patch(lat, _norm_lon(lon), half_size_m, cells)


@app.get("/api/proposals")
def proposals_list():
    return store.list_all()


@app.post("/api/proposals")
def proposals_create(req: ProposalCreate):
    return store.create(req.name, req.evaluation, req.notes)


@app.get("/api/proposals/{pid}")
def proposals_get(pid: str):
    p = store.get(pid)
    if not p:
        raise HTTPException(404, "proposal not found")
    return p


@app.patch("/api/proposals/{pid}")
def proposals_update(pid: str, req: ProposalUpdate):
    try:
        p = store.update(pid, **req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not p:
        raise HTTPException(404, "proposal not found")
    return p


@app.delete("/api/proposals/{pid}")
def proposals_delete(pid: str):
    if not store.delete(pid):
        raise HTTPException(404, "proposal not found")
    return {"ok": True}


# ------------------------------------------------------------------ AI


class AssessRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=360)
    landing_time: datetime


class RecommendRequest(BaseModel):
    sites: list[Point] = Field(min_length=1, max_length=12)
    start: datetime
    days: float = Field(30.0, gt=0, le=90)
    step_hours: float = Field(12.0, ge=1, le=72)
    divert_radius_m: float = Field(3000.0, ge=300, le=10000)
    n_backups: int = Field(4, ge=0, le=8)


@app.get("/api/ai/model-card")
def ai_model_card():
    return ai.model_card()


@app.post("/api/ai/assess")
def ai_assess(req: AssessRequest):
    return ai.assess(req.lat, _norm_lon(req.lon), req.landing_time)


@app.post("/api/ai/recommend")
def ai_recommend(req: RecommendRequest):
    sites = [{"id": s.id, "name": s.name or s.id, "lat": s.lat, "lon": _norm_lon(s.lon), "half_size_m": s.half_size_m or 7500.0} for s in req.sites]
    return regions.recommend(sites, req.start, req.days, req.step_hours)


@app.get("/api/ai/site-map")
def ai_site_map(lat: float, lon: float, landing_time: datetime, half_size_m: float = Query(3000.0, ge=500, le=10000), cells: int = Query(25, ge=5, le=61)):
    return ai.site_map(lat, _norm_lon(lon), landing_time, half_size_m, cells)


@app.get("/api/ai/timeline")
def ai_timeline(lat: float, lon: float, start: datetime, days: float = Query(30.0, gt=0, le=90), step_hours: float = Query(6.0, ge=1, le=48)):
    return ai.timeline(lat, _norm_lon(lon), start, days, step_hours)


@app.get("/api/ai/polar-map.png")
def ai_polar_map(time: datetime):
    t = time.replace(minute=0, second=0, microsecond=0)  # hourly cache granularity
    data = ai_polar.png(t.isoformat())
    if data is None:
        raise HTTPException(404, "AI polar map unavailable (model or grid not ready)")
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/regions/analyze")
def region_analyze(lat: float, lon: float, landing_time: datetime, half_size_m: float = Query(7500.0, ge=500, le=20000), cells: int = Query(61, ge=11, le=101)):
    return regions.analyze(lat, _norm_lon(lon), half_size_m, landing_time, cells)


@app.get("/api/regions/timeline")
def region_timeline(lat: float, lon: float, start: datetime, half_size_m: float = Query(7500.0, ge=500, le=20000), days: float = Query(30.0, gt=0, le=90), step_hours: float = Query(6.0, ge=1, le=48)):
    return regions.timeline(lat, _norm_lon(lon), half_size_m, start, days, step_hours)
