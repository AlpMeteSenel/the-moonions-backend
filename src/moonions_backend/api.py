"""FastAPI application."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ai, ai_polar, conditions, diskcache, ml_inference, regions, store, tiles
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


@app.on_event("startup")
def _warm_ai_maps():
    # load the AI model and polar grid off the request path, and render the maps around "now"
    ai_polar.prewarm(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1))


@app.on_event("shutdown")
def _stop_workers():
    from .regions import shutdown_pool

    shutdown_pool()
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2048)

# GET results for an explicit time (or pure terrain) never change: let browsers keep them, so
# scrubbing back and reloading the page are instant. "Now"-dependent calls are left uncached.
_TIMED = ("time", "start", "landing_time")
_IMMUTABLE_PATHS = ("/api/terrain",)
_NEVER = ("/api/agents", "/api/proposals", "/api/health")


@app.middleware("http")
async def cache_headers(request, call_next):
    resp = await call_next(request)
    path = request.url.path
    if request.method == "GET" and resp.status_code == 200 and path.startswith("/api/") and not path.startswith(_NEVER) and "cache-control" not in resp.headers:
        if path.startswith(_IMMUTABLE_PATHS) or any(k in request.query_params for k in _TIMED):
            resp.headers["Cache-Control"] = "public, max-age=86400"
        elif path == "/api/sites":
            resp.headers["Cache-Control"] = "public, max-age=300"
    return resp
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


@app.get("/api/sun/track")
def sun_track(start: datetime, hours: int = Query(32 * 24, ge=1, le=24 * 90), step_hours: float = Query(1.0, ge=0.25, le=24)):
    """Sub-solar / sub-Earth points over a time span, so the dashboard animates lighting locally."""
    key = [start.isoformat(), hours, step_hours]

    def build():
        out = []
        n = int(hours / step_hours) + 1
        for k in range(n):
            t = start + timedelta(hours=step_hours * k)
            p = sun_and_earth_points(t)
            out.append([round(p["sun_lat"], 4), round(p["sun_lon"], 4), round(p["earth_lat"], 4), round(p["earth_lon"], 4)])
        return {"start": start.isoformat(), "step_hours": step_hours, "points": out}

    return diskcache.cached_json("sun_track", key, build)


@app.post("/api/ai/prewarm")
def ai_prewarm(start: datetime, hours: int = Query(48, ge=1, le=24 * 32), step_hours: int = Query(1, ge=1, le=24)):
    """Queue the polar AI map for upcoming hours (timeline playback asks for these next)."""
    t = start.replace(minute=0, second=0, microsecond=0)
    for k in range(0, hours + 1, step_hours):
        ai_polar.prewarm_one(t + timedelta(hours=k))
    return {"queued": hours // step_hours + 1}


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


@app.get("/api/tiles/info")
def tiles_info():
    return tiles.info()


@app.get("/api/tiles/{pole}/{z}/{x}/{y}.bin")
def tile_bin(pole: Literal["s", "n"], z: int, x: int, y: int):
    """LOLA polar height tile: (128+3)^2 int16 DN (0.5 m) incl. a 1-sample border."""
    data = tiles.tile(pole, z, x, y)
    if data is None:
        raise HTTPException(404, "no such tile")
    return Response(data, media_type="application/octet-stream", headers={"Cache-Control": "public, max-age=31536000, immutable"})


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
    key = [round(lat, 6), _norm_lon(lon), start.isoformat(), days, step_hours]
    return diskcache.cached_json("ai_tl", key, lambda: ai.timeline(lat, _norm_lon(lon), start, days, step_hours))


@app.get("/api/ai/polar-map.png")
def ai_polar_map(time: datetime):
    t = time.replace(minute=0, second=0, microsecond=0)  # hourly cache granularity
    data = diskcache.cached_bytes("polar_png", t.isoformat(), lambda: ai_polar.png(t.isoformat()))
    ai_polar.prewarm(t)  # the hours either side are what the timeline asks for next
    if data is None:
        raise HTTPException(404, "AI polar map unavailable (model or grid not ready)")
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/regions/analyze")
def region_analyze(lat: float, lon: float, landing_time: datetime, half_size_m: float = Query(7500.0, ge=500, le=20000), cells: int = Query(61, ge=11, le=101)):
    return regions.analyze(lat, _norm_lon(lon), half_size_m, landing_time, cells)


@app.get("/api/regions/analyze-range")
def region_analyze_range(
    lat: float,
    lon: float,
    start: datetime,
    hours: int = Query(24, ge=1, le=96),
    step_hours: int = Query(1, ge=1, le=24),
    half_size_m: float = Query(7500.0, ge=500, le=20000),
    cells: int = Query(61, ge=11, le=101),
):
    """Region analysis for a block of hours in one request (timeline playback fetches ahead)."""
    t = start.replace(minute=0, second=0, microsecond=0)
    key = [round(lat, 6), _norm_lon(lon), half_size_m, cells, t.isoformat(), hours, step_hours]
    return diskcache.cached_json("region_range", key, lambda: regions.analyze_range(lat, _norm_lon(lon), half_size_m, t, hours, step_hours, cells))


@app.get("/api/regions/timeline")
def region_timeline(lat: float, lon: float, start: datetime, half_size_m: float = Query(7500.0, ge=500, le=20000), days: float = Query(30.0, gt=0, le=90), step_hours: float = Query(6.0, ge=1, le=48)):
    key = [round(lat, 6), _norm_lon(lon), half_size_m, start.isoformat(), days, step_hours]
    return diskcache.cached_json("region_tl", key, lambda: regions.timeline(lat, _norm_lon(lon), half_size_m, start, days, step_hours))


class SummaryRequest(BaseModel):
    regions: list[Point] = Field(min_length=1, max_length=64)
    time: datetime


@app.post("/api/regions/summary")
def regions_summary(req: SummaryRequest):
    t = req.time.replace(minute=0, second=0, microsecond=0)
    regs = [{"id": r.id, "lat": r.lat, "lon": _norm_lon(r.lon), "half_size_m": r.half_size_m} for r in req.regions]
    return {"time": t.isoformat(), "regions": ai_polar.region_summaries(regs, t.isoformat())}


@app.get("/api/conditions")
def site_conditions(
    lat: float,
    lon: float,
    start: datetime,
    days: float = Query(14.0, gt=0, le=60),
    step_min: float = Query(10.0, ge=1, le=120),
    comm_mask_deg: float = Query(1.0, ge=0, le=15),
    mission_hours: float = Query(156.0, ge=1, le=720),
):
    """Sun & Earth vs the local skyline: countdowns to sunset/Earthset, power and comms windows."""
    return conditions.site_conditions(lat, _norm_lon(lon), start, days, step_min, comm_mask_deg, mission_hours)


class AreaPlanRequest(BaseModel):
    polygon: list[tuple[float, float]] | None = Field(None, min_length=3, max_length=400, description="[[lat, lon], ...] outline drawn by the user")
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=360)
    radius_m: float | None = Field(None, ge=1000, le=60000)
    request: str = Field("", max_length=1000)
    start: datetime | None = None
    days: float = Field(30.0, ge=1, le=32)
    stay_hours: float = Field(156.0, ge=12, le=720)


def _plan_polygon(req: AreaPlanRequest) -> list[tuple[float, float]]:
    from . import planner

    if req.polygon:
        return [(float(a), _norm_lon(float(b))) for a, b in req.polygon]
    if req.lat is None or req.lon is None or req.radius_m is None:
        raise HTTPException(422, "send a polygon, or lat/lon/radius_m")
    return planner.circle_polygon(req.lat, _norm_lon(req.lon), req.radius_m)


@app.post("/api/plan/area")
def plan_area(req: AreaPlanRequest):
    """Circle + goal -> an LLM agent researches the area with the backend's tools and proposes landing spots (NDJSON stream)."""
    from fastapi.responses import StreamingResponse

    from . import planner

    return StreamingResponse(
        planner.stream(_plan_polygon(req), req.request, req.start, req.days, req.stay_hours),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/agents/run")
def agents_run(start: datetime, days: float = Query(14.0, ge=1, le=30), stay_hours: float = Query(156.0, ge=12, le=720)):
    """Run the mission-control agent team; streams one JSON event per line as the agents work."""
    from fastapi.responses import StreamingResponse

    from . import agents

    return StreamingResponse(agents.stream(start, days, stay_hours), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
