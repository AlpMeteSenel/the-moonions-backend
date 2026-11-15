"""Area planner: the user circles an area and says what they want to do; an LLM agent researches it
with this backend's real tools and proposes where (and when) to land.

The agent never invents numbers: every figure comes from a tool (terrain, sunlight/Earth/thermal
model, AI landing timeline and zones). Its proposals are re-checked against the tools before they
are shown. Without an LLM (no key, MOONIONS_LLM=0, or an error) a rule-based planner answers.

Streams NDJSON events: status, candidates, tool, tool_result, think, proposals, error, done.
Self-contained on purpose (one module + one endpoint): easy to remove.
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

import numpy as np

from .agents import _llm_config
from .ephemeris import to_utc

MAX_ROUNDS = 12
MAX_CANDIDATES = 16


# ----------------------------------------------------------------------------- the drawn area
class Area:
    """The user's drawn polygon in a local east/north frame (metres) around its centroid."""

    def __init__(self, polygon: list[tuple[float, float]]):
        from .dem import distance_bearing

        v = np.array([[math.cos(math.radians(a)) * math.cos(math.radians(b)), math.cos(math.radians(a)) * math.sin(math.radians(b)), math.sin(math.radians(a))] for a, b in polygon])
        c = v.mean(0)
        c /= np.linalg.norm(c)
        self.lat, self.lon = math.degrees(math.asin(c[2])), math.degrees(math.atan2(c[1], c[0]))
        en = []
        for a, b in polygon:
            d, brg = distance_bearing(self.lat, self.lon, a, b)
            en.append((float(d) * math.sin(math.radians(float(brg))), float(d) * math.cos(math.radians(float(brg)))))
        self.en = np.array(en)
        self.radius_m = float(np.hypot(self.en[:, 0], self.en[:, 1]).max())
        x, y = self.en[:, 0], self.en[:, 1]
        self.area_km2 = abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2e6

    def inside(self, e: float, n: float) -> bool:
        x, y = self.en[:, 0], self.en[:, 1]
        hit = False
        j = len(x) - 1
        for i in range(len(x)):
            if (y[i] > n) != (y[j] > n) and e < (x[j] - x[i]) * (n - y[i]) / (y[j] - y[i] + 1e-12) + x[i]:
                hit = not hit
            j = i
        return hit


def _candidates(area: Area) -> tuple[list[dict], float]:
    """Square candidate landing areas tiling the drawn area (half-size scales with its size)."""
    from .dem import offset_latlon

    lat, lon = area.lat, area.lon
    half = float(np.clip(math.sqrt(max(area.area_km2, 1.0)) * 1000 / 6.0, 1500.0, 5000.0))
    while True:
        step = 2 * half
        n = int(area.radius_m // step) + 1
        pts = [(e * step, nn * step) for e in range(-n, n + 1) for nn in range(-n, n + 1) if area.inside(e * step, nn * step)]
        if len(pts) <= MAX_CANDIDATES or half >= 5000:
            break
        half *= 1.25
    if not pts:
        pts = [(0.0, 0.0)]
    out = []
    for k, (e, nn) in enumerate(sorted(pts, key=lambda p: math.hypot(*p))):
        la, lo = offset_latlon(lat, lon, np.array([e]), np.array([nn]))
        out.append({"id": f"C{k + 1}", "lat": round(float(la[0]), 5), "lon": round(float(lo[0]), 5), "east_m": e, "north_m": nn})
    return out[:MAX_CANDIDATES], half


def _compass(east: float, north: float) -> str:
    from .backups import compass

    if abs(east) < 1 and abs(north) < 1:
        return "centre"
    return compass(math.degrees(math.atan2(east, north)) % 360)


# ----------------------------------------------------------------------------- the tools
class Tools:
    def __init__(self, area: Area, start: datetime, days: float, stay_hours: float):
        self.area = area
        self.lat, self.lon, self.radius_m = area.lat, area.lon, area.radius_m
        self.start, self.days, self.stay_hours = to_utc(start), days, stay_hours
        self.cands, self.half = _candidates(area)
        self.by_id = {c["id"]: c for c in self.cands}

    def _c(self, cid: str) -> dict:
        if cid not in self.by_id:
            raise ValueError(f"unknown candidate {cid}; use one of {', '.join(self.by_id)}")
        return self.by_id[cid]

    def _t(self, iso: str) -> datetime:
        t = to_utc(datetime.fromisoformat(iso.replace("Z", "+00:00"))).replace(minute=0, second=0, microsecond=0)
        if not (self.start <= t <= self.start + timedelta(days=self.days)):
            raise ValueError(f"time outside the planning window {self.start.isoformat()} + {self.days:g} days")
        return t

    def list_candidates(self) -> dict:
        from .thermal import slope_aspect
        from .dem import get_dem

        dem = get_dem()
        rows = []
        for c in self.cands:
            s, a = slope_aspect(c["lat"], c["lon"])
            rows.append(
                {
                    "id": c["id"],
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "from_area_centre": f"{math.hypot(c['east_m'], c['north_m']) / 1000:.1f} km {_compass(c['east_m'], c['north_m'])}",
                    "elevation_m": round(float(dem.elevation(np.array(c["lat"]), np.array(c["lon"]))), 0),
                    "centre_slope_deg": round(s, 1),
                    "faces_deg": round(a, 0),
                }
            )
        return {"candidate_size_km": round(2 * self.half / 1000, 1), "candidates": rows}

    def survey(self, candidate_ids: list[str] | None = None) -> dict:
        """Sunlight, Earth visibility and surface temperature over the window at each candidate centre."""
        from .conditions import visibility
        from .thermal import site_temperature

        step_h = 3.0
        times = [self.start + timedelta(hours=step_h * k) for k in range(int(self.days * 24 / step_h) + 1)]
        rows = []
        for cid in candidate_ids or list(self.by_id):
            c = self._c(cid)
            v = visibility(c["lat"], c["lon"], times)
            th = site_temperature(c["lat"], c["lon"], times, v, step_h)
            lit = np.asarray(v["lit"], bool)
            dark = longest = 0
            for x in lit:
                dark = 0 if x else dark + 1
                longest = max(longest, dark)
            rows.append(
                {
                    "id": cid,
                    "sunlit_share": round(float(lit.mean()), 3),
                    "earth_in_view_share": round(float(np.asarray(v["earth_ok"]).mean()), 3),
                    "longest_darkness_h": round(longest * step_h, 0),
                    "surface_temp_min_c": round(float(th["temp_k"].min()) - 273.15, 0),
                    "surface_temp_max_c": round(float(th["temp_k"].max()) - 273.15, 0),
                }
            )
        return {"window_days": self.days, "candidates": rows}

    def cold_traps(self) -> dict:
        """Points in the drawn area that never see the Sun in the window (likely volatiles / ice), with temperatures."""
        from .conditions import visibility
        from .dem import distance_bearing, offset_latlon
        from .thermal import site_temperature

        step_h = 6.0
        times = [self.start + timedelta(hours=step_h * k) for k in range(int(self.days * 24 / step_h) + 1)]
        n = 8
        ax = np.linspace(-self.radius_m, self.radius_m, n)
        pts = [(e, nn) for e in ax for nn in ax if self.area.inside(e, nn)] or [(0.0, 0.0)]
        traps = []
        for e, nn in pts:
            la, lo = offset_latlon(self.lat, self.lon, np.array([e]), np.array([nn]))
            la, lo = float(la[0]), float(lo[0])
            v = visibility(la, lo, times)
            share = float(np.asarray(v["sun_frac"]).mean())
            if share <= 0.02:
                th = site_temperature(la, lo, times, v, step_h)
                near = min(self.cands, key=lambda c: distance_bearing(la, lo, c["lat"], c["lon"])[0])
                d = float(distance_bearing(la, lo, near["lat"], near["lon"])[0])
                traps.append({"lat": round(la, 4), "lon": round(lo, 4), "sun_share": round(share, 3), "max_temp_c": round(float(th["temp_k"].max()) - 273.15, 0), "nearest_candidate": near["id"], "distance_km": round(d / 1000, 1)})
            if len(traps) >= 8:
                break
        traps.sort(key=lambda x: x["max_temp_c"])
        return {"sampled_points": len(pts), "cold_traps": traps}

    def landing_timeline(self, candidate_id: str) -> dict:
        from . import regions

        c = self._c(candidate_id)
        tl = regions.timeline(c["lat"], c["lon"], self.half, self.start, self.days, 6)
        if not tl.get("available"):
            return {"available": False}
        vs = tl["viable_share"]
        k = int(np.argmax(tl["score"]))
        return {
            "id": candidate_id,
            "best_time": tl["times"][k],
            "viable_share_at_best": vs[k],
            "best_spot_success_at_best": tl["best_success"][k],
            "share_of_window_with_any_viable_ground": round(float(np.mean(np.asarray(vs) > 0.02)), 3),
            "windows": [{"start": w["start"], "end": w["end"], "best": w["best"], "best_success": w.get("best_success")} for w in tl["windows"][:3]],
        }

    def landing_zones(self, candidate_id: str, time_iso: str) -> dict:
        from . import regions

        c = self._c(candidate_id)
        a = regions.analyze(c["lat"], c["lon"], self.half, self._t(time_iso), 61)
        if not a.get("available"):
            return {"available": False}
        st = a["stats"]
        return {
            "id": candidate_id,
            "time": a["landing_time"],
            "viable_share": st["viable_share"],
            "viable_km2": st["viable_km2"],
            "lit_share": st["lit_share"],
            "power_share": st["power_share"],
            "earth_link_share": st["comms_share"],
            "safe_ground_share": st["safe_ground_share"],
            "zones": [
                {k: z[k] for k in ("id", "lat", "lon", "area_km2", "mean_success", "margin_m") if k in z} | ({"from_primary": z["from_primary"]} if z.get("from_primary") else {})
                for z in a["zones"][:4]
            ],
        }

    def stay_conditions(self, candidate_id: str, time_iso: str) -> dict:
        from .conditions import site_conditions

        c = self._c(candidate_id)
        cd = site_conditions(c["lat"], c["lon"], self._t(time_iso), 14.0, mission_hours=self.stay_hours)
        th = cd.get("thermal") or {}
        return {
            "id": candidate_id,
            "sun_up_at_touchdown": cd["now"]["sun_up"],
            "sun_elevation_deg": cd["now"]["sun_elevation_deg"],
            "earth_in_view_at_touchdown": cd["now"]["earth_up"],
            "next_sunset_in_h": (cd["next"]["sunset"] or {}).get("in_hours"),
            "stay_hours": cd["stay"]["hours"],
            "stay_power_ok": cd["stay"]["power_ok"],
            "stay_lowest_battery": cd["stay"]["lowest_battery"],
            "stay_sunlit_share": cd["stay"]["sunlit_share"],
            "stay_earth_share": cd["stay"]["earth_share"],
            "surface_temp_c": {k.replace("_k", ""): round(v - 273.15, 0) for k, v in th.items() if k in ("now_k", "stay_min_k", "stay_max_k")},
        }


TOOL_SPECS = [
    {"name": "list_candidates", "description": "The candidate landing areas tiling the drawn area: position, elevation, slope at the centre.", "parameters": {"type": "object", "properties": {}}},
    {
        "name": "survey",
        "description": "Sunlit share, Earth-in-view share, longest darkness and min/max surface temperature over the planning window at candidate centres (all if no ids).",
        "parameters": {"type": "object", "properties": {"candidate_ids": {"type": "array", "items": {"type": "string"}}}},
    },
    {"name": "cold_traps", "description": "Sample the drawn area for ground that never sees the Sun in the window (likely ice/volatiles): location, peak temperature, nearest candidate.", "parameters": {"type": "object", "properties": {}}},
    {
        "name": "landing_timeline",
        "description": "AI landing-success timeline of one candidate area over the window: best time, viable share of the area, landing windows.",
        "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string"}}, "required": ["candidate_id"]},
    },
    {
        "name": "landing_zones",
        "description": "Landing zones (Z1 primary + backups that fit the landing ellipse) and area statistics of a candidate at one touchdown time (ISO, within the window).",
        "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string"}, "time_iso": {"type": "string"}}, "required": ["candidate_id", "time_iso"]},
    },
    {
        "name": "stay_conditions",
        "description": "Sun, Earth, battery and surface temperature over the surface stay for a touchdown at a candidate centre at a time (ISO).",
        "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string"}, "time_iso": {"type": "string"}}, "required": ["candidate_id", "time_iso"]},
    },
    {
        "name": "submit_proposals",
        "description": "Finish: 1 to 3 proposals, best first. Each must have been checked with landing_zones at its time.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-3 sentences for the user."},
                "proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "time_iso": {"type": "string"},
                            "zone_id": {"type": "string", "description": "Z1, Z2, ... from landing_zones"},
                            "title": {"type": "string", "description": "short name, e.g. 'Ridge near the cold trap'"},
                            "rationale": {"type": "string", "description": "why this fits the user's goal, citing tool numbers"},
                            "risks": {"type": "string"},
                        },
                        "required": ["candidate_id", "time_iso", "title", "rationale"],
                    },
                },
            },
            "required": ["summary", "proposals"],
        },
    },
]

SYSTEM = """You are the mission planner of a lunar landing dashboard (NASA LOLA terrain, JPL ephemerides, a trained AI landing-success model).
The user outlined an area of the Moon and described what they want to do there. Research the area with the tools and propose where and when to land.

Rules:
- Use only numbers returned by the tools; never invent figures.
- Match the goal: e.g. ice/volatiles -> land in sunlight close to a cold trap; long stay / power -> high sunlit share and short darkness; communications -> Earth in view; safety -> large viable area and backup zones.
- A good touchdown needs the Sun up, Earth in view and power through the stay; landing_zones combines these with AI touchdown safety.
- Be efficient: survey or list first, check landing_timeline for the 2-4 most promising candidates, then landing_zones (and stay_conditions if relevant) at their best times.
- Before each tool call, say in one short sentence what you are checking and why.
- Finish with submit_proposals (1-3 proposals, best first)."""


# ----------------------------------------------------------------------------- LLM + loop
def _chat(cfg: dict, messages: list[dict], tool_choice: Any = "auto") -> dict:
    body = {"model": cfg["model"], "messages": messages, "tools": [{"type": "function", "function": t} for t in TOOL_SPECS], "tool_choice": tool_choice}
    req = urllib.request.Request(
        cfg["base"] + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json", "X-Title": "Plan Banana lunar landing navigator"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]


def _brief(name: str, res: dict) -> str:
    """One line for the UI log."""
    if name == "list_candidates":
        return f"{len(res['candidates'])} candidate areas of {res['candidate_size_km']} km"
    if name == "survey":
        best = max(res["candidates"], key=lambda r: r["sunlit_share"])
        return f"{len(res['candidates'])} surveyed; most sunlight {best['id']} ({best['sunlit_share']:.0%})"
    if name == "cold_traps":
        t = res["cold_traps"]
        return f"{len(t)} cold traps found" + (f"; coldest peaks at {t[0]['max_temp_c']:.0f} °C near {t[0]['nearest_candidate']}" if t else "")
    if name == "landing_timeline":
        return f"{res.get('id')}: best {res.get('best_time', '?')[:16]} with {res.get('viable_share_at_best', 0):.0%} viable" if res.get("available", True) else "no AI timeline"
    if name == "landing_zones":
        return f"{res.get('id')} at {res.get('time', '')[:16]}: {res.get('viable_share', 0):.0%} viable, {len(res.get('zones', []))} zones"
    if name == "stay_conditions":
        return f"{res['id']}: power through the stay {'OK' if res['stay_power_ok'] else 'FAILS'}, Earth {res['stay_earth_share']:.0%} of the stay"
    return ""


def _finalise(tools: Tools, raw: list[dict]) -> list[dict]:
    """Re-check every proposal with the tools and attach the zone the dashboard will show."""
    out = []
    for p in raw[:3]:
        try:
            z = tools.landing_zones(p["candidate_id"], p["time_iso"])
        except Exception:
            continue
        if not z.get("available", True) or not z.get("zones"):
            continue
        zone = next((x for x in z["zones"] if x["id"] == p.get("zone_id")), z["zones"][0])
        c = tools.by_id[p["candidate_id"]]
        out.append(
            {
                "candidate_id": p["candidate_id"],
                "title": p.get("title") or f"{p['candidate_id']} {zone['id']}",
                "time": z["time"],
                "area": {"lat": c["lat"], "lon": c["lon"], "half_size_m": tools.half},
                "zone": zone,
                "viable_share": z["viable_share"],
                "viable_km2": z["viable_km2"],
                "rationale": p.get("rationale", ""),
                "risks": p.get("risks", ""),
            }
        )
    return out


def _rule_based(tools: Tools, emit) -> tuple[list[dict], str]:
    emit("status", "No language model available: ranking candidates by their AI landing timelines.")
    scored = []
    for c in tools.cands:
        tl = tools.landing_timeline(c["id"])
        emit("tool_result", _brief("landing_timeline", tl), name="landing_timeline")
        if tl.get("available", True) and tl.get("best_time"):
            scored.append((tl["viable_share_at_best"] * tl["best_spot_success_at_best"], c["id"], tl["best_time"]))
    scored.sort(reverse=True)
    props = [{"candidate_id": cid, "time_iso": t, "title": f"{cid}: best AI landing window", "rationale": f"Largest predicted landable share in the window ({s:.0%} weighted)."} for s, cid, t in scored[:3]]
    return props, "Ranked by the AI landing timeline only (no language model)."


def plan(polygon: list[tuple[float, float]], request: str, start: datetime, days: float, stay_hours: float, emit) -> None:
    area = Area(polygon)
    tools = Tools(area, start, days, stay_hours)
    lat, lon = area.lat, area.lon
    emit("candidates", f"Split the {area.area_km2:.0f} km² area into {len(tools.cands)} candidate areas of {2 * tools.half / 1000:.1f} km.", candidates=[{"id": c["id"], "lat": c["lat"], "lon": c["lon"], "half_size_m": tools.half} for c in tools.cands])
    cfg = _llm_config()
    summary = ""
    raw: list[dict] = []
    if cfg is None:
        raw, summary = _rule_based(tools, emit)
    else:
        emit("status", f"{cfg['model']} is researching the area…", llm=cfg["model"])
        user = (
            f"Goal: {request.strip() or 'find the best place to land'}\n"
            f"Area: {area.area_km2:.0f} km² polygon around {lat:.4f}, {lon:.4f} (up to {area.radius_m / 1000:.1f} km from its centre). Planning window: {tools.start.isoformat()} + {days:g} days. "
            f"Surface stay: {stay_hours:g} h.\nCandidates: " + json.dumps(tools.list_candidates()["candidates"])
        )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        for rnd in range(MAX_ROUNDS):
            force = rnd == MAX_ROUNDS - 1
            try:
                msg = _chat(cfg, messages, {"type": "function", "function": {"name": "submit_proposals"}} if force else "auto")
            except Exception as exc:  # noqa: BLE001
                emit("error", f"Language model unavailable ({exc}); falling back to the rule-based planner.")
                raw, summary = _rule_based(tools, emit)
                break
            if msg.get("content"):
                emit("think", msg["content"].strip())
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls else {"role": "assistant", "content": msg.get("content") or ""})
            if not calls:
                messages.append({"role": "user", "content": "Continue with the tools, then call submit_proposals."})
                continue
            done = False
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                if name == "submit_proposals":
                    raw, summary, done = args.get("proposals", []), args.get("summary", ""), True
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": "ok"})
                    continue
                emit("tool", f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})", name=name)
                t0 = time.time()
                try:
                    res = getattr(tools, name)(**args)
                    emit("tool_result", _brief(name, res), name=name, seconds=round(time.time() - t0, 1))
                    content = json.dumps(res, separators=(",", ":"))[:6000]
                except Exception as exc:  # noqa: BLE001
                    content = json.dumps({"error": str(exc)})
                    emit("tool_result", f"{name} failed: {exc}", name=name)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            if done:
                break
    proposals = _finalise(tools, raw)
    if not proposals:
        emit("status", "No proposal survived the check; trying the rule-based ranking.")
        raw, summary = _rule_based(tools, emit)
        proposals = _finalise(tools, raw)
    emit("proposals", summary or "Proposals ready.", proposals=proposals)


def circle_polygon(lat: float, lon: float, radius_m: float, n: int = 32) -> list[tuple[float, float]]:
    from .dem import destination

    out = []
    for k in range(n):
        a, b = destination(lat, lon, 360 * k / n, radius_m)
        out.append((float(a), float(b)))
    return out


async def stream(polygon: list[tuple[float, float]], request: str, start: datetime | None, days: float, stay_hours: float) -> AsyncIterator[str]:
    q: queue.Queue = queue.Queue()
    t_start = time.time()

    def emit(kind: str, text: str, **extra) -> None:
        q.put({"t": round(time.time() - t_start, 1), "kind": kind, "text": text, **extra})

    def run() -> None:
        try:
            plan(polygon, request, start or datetime.now(timezone.utc), days, stay_hours, emit)
        except Exception as exc:  # noqa: BLE001
            emit("error", f"Planner failed: {exc}")
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True, name="area-planner").start()
    while True:
        ev = await asyncio.to_thread(q.get)
        if ev is None:
            break
        yield json.dumps(ev) + "\n"
    yield json.dumps({"t": round(time.time() - t_start, 1), "kind": "done", "text": ""}) + "\n"
