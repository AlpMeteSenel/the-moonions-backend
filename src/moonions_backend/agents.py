"""Mission-control agent team: decides where and when to land, streaming what each agent does.

Seven agents with distinct jobs, each using the real tools of this backend (no invented numbers):

  Chief  (Flight Director)   plans, dispatches, resolves vetoes, makes the call (LLM if available)
  Ada    (Site Scout)        screens every candidate region over the whole window   -> shortlist
  Rocco  (Terrain)           AI landing-success map of a finalist at its best time  -> zones
  Sol    (Power)             Sun vs skyline + battery over the stay at the zone     -> vote
  Echo   (Comms)             Earth vs skyline over the stay at the zone             -> vote
  Tempo  (Timing)            how long the window stays open, when it reopens        -> vote / re-time
  Nova   (Safety)            backup zones inside the region and divert distances    -> vote

Flow: Chief briefs -> Ada scouts -> finalists fan out to Rocco, then Sol + Echo (in parallel, at the
primary zone Rocco found), Tempo and Nova -> votes -> if the best finalist is vetoed on light/power/
comms, Chief asks Tempo to re-time it and Sol + Echo re-check -> Chief decides primary + backups.

The run is streamed as JSON lines (one event per line) so the UI can animate the team.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from .ephemeris import to_utc
from .sites import SITES

ROSTER = [
    {"id": "chief", "name": "Chief", "role": "Flight Director", "color": "#f472b6"},
    {"id": "scout", "name": "Ada", "role": "Site Scout", "color": "#22d3ee"},
    {"id": "terrain", "name": "Rocco", "role": "Terrain", "color": "#a3e635"},
    {"id": "power", "name": "Sol", "role": "Power", "color": "#fbbf24"},
    {"id": "comms", "name": "Echo", "role": "Comms", "color": "#60a5fa"},
    {"id": "timing", "name": "Tempo", "role": "Timing", "color": "#c084fc"},
    {"id": "safety", "name": "Nova", "role": "Safety", "color": "#fb7185"},
]
FINALISTS = 3
MIN_SCORE = 0.05  # a region-time below this has essentially no landable area
VOTE_POINTS = {"GO": 2, "CAUTION": 1, "NO-GO": 0}

_pool = ThreadPoolExecutor(max_workers=6)


# ----------------------------------------------------------------------------- LLM (optional)
def _llm_config() -> dict | None:
    """OpenRouter by default (OPENROUTER_API_KEY in the backend's .env); any OpenAI-compatible
    endpoint can be used instead via MOONIONS_LLM_BASE_URL + MOONIONS_LLM_API_KEY."""
    from . import config  # noqa: F401  (loads .env)

    if os.environ.get("MOONIONS_LLM", "1") == "0":
        return None
    key = os.environ.get("MOONIONS_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    base = os.environ.get("MOONIONS_LLM_BASE_URL", "https://openrouter.ai/api/v1")
    return {"base": base.rstrip("/"), "key": key, "model": os.environ.get("MOONIONS_LLM_MODEL", "openai/gpt-6-luna")}


def _llm_json(system: str, user: str, timeout: float = 45.0) -> dict | None:
    cfg = _llm_config()
    if cfg is None:
        return None
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        cfg["base"] + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {cfg['key']}", "Content-Type": "application/json", "X-Title": "Moonions lunar landing navigator"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
        text = text[text.find("{") : text.rfind("}") + 1]
        return json.loads(text)
    except Exception:
        return None


def llm_model() -> str | None:
    cfg = _llm_config()
    return cfg["model"] if cfg else None


# ----------------------------------------------------------------------------- formatting helpers
def _fmt_t(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d %b %H:%M UTC")


def _fmt_h(h: float) -> str:
    if h < 24:
        return f"{h:.0f} h"
    d, r = divmod(round(h), 24)
    return f"{d} d {r} h" if r else f"{d} d"


# ----------------------------------------------------------------------------- the team
class Mission:
    def __init__(self, start: datetime, days: float, stay_hours: float):
        self.start = to_utc(start)
        self.days = days
        self.stay_hours = stay_hours
        self.q: asyncio.Queue[dict | None] = asyncio.Queue()
        self.t0 = time.monotonic()
        self.seq = 0

    # -- events
    def emit(self, agent: str, kind: str, text: str, **extra) -> None:
        self.seq += 1
        self.q.put_nowait({"seq": self.seq, "t": round(time.monotonic() - self.t0, 2), "agent": agent, "kind": kind, "text": text, **extra})

    async def tool(self, agent: str, label: str, fn, *args):
        self.emit(agent, "tool", label)
        t = time.monotonic()
        out = await asyncio.get_running_loop().run_in_executor(_pool, lambda: fn(*args))
        self.emit(agent, "tool_done", label, ms=round((time.monotonic() - t) * 1000))
        return out

    # -- agents
    async def scout(self, start: datetime, days: float) -> list[dict]:
        from .regions import timeline

        regs = [s for s in SITES if s.get("group") == "Artemis III region"]
        self.emit("scout", "say", f"Screening all {len(regs)} Artemis III regions from {start:%d %b} over {days:g} days, every 6 h.")
        tls = await asyncio.gather(
            *[self.tool("scout", f"Landing-success timeline · {r['name']}", timeline, r["lat"], r["lon"], r["half_size_m"], start, days, 6.0) for r in regs]
        )
        ranked = []
        for r, tl in zip(regs, tls):
            if not tl.get("available"):
                continue
            k = max(range(len(tl["score"])), key=lambda i: tl["score"][i])
            ranked.append({"site": r, "timeline": tl, "best_time": tl["times"][k], "score": tl["score"][k], "viable_share": tl["viable_share"][k], "best_success": tl["best_success"][k]})
        ranked.sort(key=lambda x: -x["score"])
        top = [x for x in ranked if x["score"] >= MIN_SCORE][:FINALISTS]
        dropped = [x["site"]["name"] for x in ranked if x["score"] < MIN_SCORE]
        if not top:
            self.emit("scout", "report", f"Nothing is landable anywhere between {start:%d %b} and {start + timedelta(days=days):%d %b}: the regions are dark or have no ground that fits the ellipse.", data={"shortlist": []})
            return []
        self.emit(
            "scout",
            "report",
            "Shortlist: " + "; ".join(f"{x['site']['name']} at {_fmt_t(x['best_time'])} ({x['viable_share']*100:.0f}% of area landable)" for x in top)
            + (f". Ruled out: {', '.join(dropped)}." if dropped else "."),
            data={"shortlist": [{"site_id": x["site"]["id"], "name": x["site"]["name"], "time": x["best_time"], "score": x["score"], "viable_share": x["viable_share"]} for x in ranked]},
        )
        return top

    async def terrain(self, f: dict) -> dict:
        from .regions import analyze

        s = f["site"]
        a = await self.tool("terrain", f"AI landing map · {s['name']} · {_fmt_t(f['time'])}", analyze, s["lat"], s["lon"], s["half_size_m"], datetime.fromisoformat(f["time"]), 61)
        st, zones = a["stats"], a["zones"]
        f["analysis"] = a
        f["zone"] = zones[0] if zones else None
        if not zones:
            vote, why = "NO-GO", f"no zone fits the ±{a['ellipse_radius_m']:.0f} m landing ellipse"
        elif st["viable_share"] < 0.1:
            vote, why = "CAUTION", f"only {st['viable_share']*100:.0f}% of the region is landable ({st['viable_km2']:.0f} km²)"
        else:
            vote, why = "GO", f"{st['viable_km2']:.0f} km² landable ({st['viable_share']*100:.0f}%), Z1 {zones[0]['area_km2']:.0f} km² at {zones[0]['mean_success']*100:.0f}% success"
        return self._vote("terrain", f, vote, why)

    async def power(self, f: dict) -> dict:
        from .conditions import site_conditions

        lat, lon = self._where(f)
        c = await self.tool("power", f"Sun vs skyline + battery · {f['site']['name']}", site_conditions, lat, lon, datetime.fromisoformat(f["time"]), 14.0, 10.0, 1.0, self.stay_hours)
        f["conditions"] = c
        st, nx = c["stay"], c["next"]
        if not c["now"]["sun_up"]:
            vote, why = "NO-GO", f"Sun behind the terrain at touchdown" + (f", rises in {_fmt_h(nx['sunrise']['in_hours'])}" if nx["sunrise"] else "")
        elif not st["power_ok"]:
            vote, why = "NO-GO", "batteries run flat during the stay" + (f" (sunset in {_fmt_h(nx['sunset']['in_hours'])})" if nx["sunset"] else "")
        elif st["lowest_battery"] < 0.3:
            vote, why = "CAUTION", f"battery dips to {st['lowest_battery']*100:.0f}%"
        else:
            sunset = f"sunset in {_fmt_h(nx['sunset']['in_hours'])}" if nx["sunset"] else "no sunset for 14 days"
            vote, why = "GO", f"Sun {c['now']['sun_above_skyline_deg']:.1f}° above the skyline, {sunset}, battery ≥ {st['lowest_battery']*100:.0f}%"
        return self._vote("power", f, vote, why)

    async def comms(self, f: dict) -> dict:
        from .conditions import site_conditions

        lat, lon = self._where(f)
        c = f.get("conditions") or await self.tool("comms", f"Earth vs skyline · {f['site']['name']}", site_conditions, lat, lon, datetime.fromisoformat(f["time"]), 14.0, 10.0, 1.0, self.stay_hours)
        if f.get("conditions"):
            self.emit("comms", "tool", f"Earth vs skyline · {f['site']['name']}")
            self.emit("comms", "tool_done", f"Earth vs skyline · {f['site']['name']}", ms=0)
        share, nx = c["stay"]["earth_share"], c["next"]
        if share < 0.5:
            vote, why = "NO-GO", f"Earth in view only {share*100:.0f}% of the stay"
        elif share < 0.999:
            vote, why = "CAUTION", f"Earth hidden {(1-share)*100:.0f}% of the stay" + (f", sets in {_fmt_h(nx['earthset']['in_hours'])}" if nx["earthset"] else "")
        else:
            vote, why = "GO", f"Earth {c['now']['earth_above_skyline_deg']:.1f}° above the skyline for the whole stay"
        return self._vote("comms", f, vote, why)

    async def timing(self, f: dict) -> dict:
        tl, t = f["timeline"], f["time"]
        self.emit("timing", "tool", f"Window length + slip options · {f['site']['name']}")
        win = next((w for w in tl["windows"] if w["start"] <= t <= w["end"]), None)
        later = [w for w in tl["windows"] if w["start"] > t]
        self.emit("timing", "tool_done", f"Window length + slip options · {f['site']['name']}", ms=0)
        left = (datetime.fromisoformat(win["end"]) - datetime.fromisoformat(t)).total_seconds() / 3600 if win else 0.0
        f["window_hours_left"] = left
        f["reopens"] = later[0]["best"] if later else None
        if win is None and tl["score"][tl["times"].index(t)] < MIN_SCORE:
            vote, why = "NO-GO", "no landing window here in this period"
        elif win is None:
            vote, why = "CAUTION", "isolated good slot, no lasting window"
        elif left >= 24:
            vote, why = "GO", f"window stays open {_fmt_h(left)} after touchdown" + (f", reopens {_fmt_t(later[0]['best'])}" if later else "")
        else:
            vote, why = "CAUTION", f"window closes {_fmt_h(left)} after touchdown: little slip tolerance" + (f", next {_fmt_t(later[0]['best'])}" if later else "")
        return self._vote("timing", f, vote, why)

    async def safety(self, f: dict) -> dict:
        zones = (f.get("analysis") or {}).get("zones") or []
        self.emit("safety", "tool", f"Backup zones + divert distances · {f['site']['name']}")
        self.emit("safety", "tool_done", f"Backup zones + divert distances · {f['site']['name']}", ms=0)
        backs = zones[1:]
        if not zones:
            vote, why = "NO-GO", "no primary zone, nothing to divert to"
        elif not backs:
            vote, why = "CAUTION", "no backup zone inside the region"
        else:
            near = min(backs, key=lambda z: z["from_primary"]["distance_m"])
            vote, why = "GO", f"{len(backs)} backup zone{'s' if len(backs) > 1 else ''}, nearest {near['from_primary']['distance_m']/1000:.1f} km {near['from_primary']['direction']}, decide by T−{near['from_primary']['decide_by_s']:.0f} s"
        return self._vote("safety", f, vote, why)

    def _where(self, f: dict) -> tuple[float, float]:
        z = f.get("zone")
        return (z["lat"], z["lon"]) if z else (f["site"]["lat"], f["site"]["lon"])

    def _vote(self, agent: str, f: dict, vote: str, why: str) -> dict:
        f.setdefault("votes", {})[agent] = {"vote": vote, "why": why}
        self.emit(agent, "vote", f"{f['site']['name']}: {vote}, {why}.", to="chief", vote=vote, site_id=f["site"]["id"])
        return f

    async def review(self, f: dict) -> dict:
        """One finalist through the specialists: terrain first (it finds the zone), then the rest in parallel."""
        self.emit("chief", "handoff", f"{f['site']['name']} at {_fmt_t(f['time'])}: Rocco, map it.", to="terrain", site_id=f["site"]["id"])
        await self.terrain(f)
        await self.power(f)  # Sol computes the Sun/Earth conditions at Z1 once; Echo reads the same run
        await asyncio.gather(self.comms(f), self.timing(f), self.safety(f))
        return f

    @staticmethod
    def tally(f: dict) -> tuple[int, list[str]]:
        vetoes = [a for a, v in f["votes"].items() if v["vote"] == "NO-GO"]
        return sum(VOTE_POINTS[v["vote"]] for v in f["votes"].values()), vetoes

    async def retime(self, f: dict) -> dict | None:
        """Chief asks Tempo for the next good slot of a vetoed finalist; Sol and Echo re-check it."""
        tl = f["timeline"]
        cands = sorted(
            (i for i, t in enumerate(tl["times"]) if t > f["time"] and tl["viable_share"][i] > 0.02),
            key=lambda i: -tl["score"][i],
        )[:3]
        self.emit("timing", "say", f"Next good slots for {f['site']['name']}: " + (", ".join(_fmt_t(tl["times"][i]) for i in cands) or "none in the window") + ".", to="chief")
        for i in cands:
            g = {**f, "time": tl["times"][i], "votes": {}, "conditions": None}
            g.pop("analysis", None)
            self.emit("chief", "handoff", f"Re-check {f['site']['name']} at {_fmt_t(g['time'])}.", to="terrain", site_id=f["site"]["id"])
            await self.review(g)
            _, vetoes = self.tally(g)
            if not vetoes:
                return g
        return None

    async def decide(self, finals: list[dict]) -> dict:
        def summary(f):
            pts, vetoes = self.tally(f)
            st = f["analysis"]["stats"]
            return {
                "option": f"{f['site']['id']}@{f['time'][:13]}", "site_id": f["site"]["id"], "name": f["site"]["name"], "time": f["time"], "points": pts, "vetoes": vetoes,
                "viable_km2": st["viable_km2"], "viable_share": st["viable_share"], "best_success": st["best_success"],
                "votes": {a: v["vote"] + ": " + v["why"] for a, v in f["votes"].items()},
            }

        table = [summary(f) for f in finals]
        ok = [t for t in table if not t["vetoes"]]
        rule = max(ok or table, key=lambda t: (t["points"], t["viable_km2"]))
        pick, rationale, concerns, by = rule, None, [], "rules"
        model = llm_model()
        if model:
            self.emit("chief", "think", f"Weighing {len(table)} options with {model}…")
            ans = await asyncio.get_running_loop().run_in_executor(
                _pool,
                lambda: _llm_json(
                    "You are the Flight Director of a crewed lunar landing (Artemis-style, south pole). Your specialists "
                    "voted on candidate landing regions and times using validated physics and an AI landing-success model. "
                    "Pick the primary and an ordered list of backups. A NO-GO vote from any specialist is a hard veto for "
                    "the primary. Prefer more landable area, more slip tolerance and nearby backup zones. Answer as JSON: "
                    '{"primary_option": str, "backup_options": [str], "rationale": "2-3 plain sentences a mission '
                    'audience understands, citing the numbers", "concerns": ["short risk to watch", ...]}',
                    json.dumps({"stay_hours": self.stay_hours, "candidates": table}),
                ),
            )
            chosen = next((t for t in table if ans and t["option"] == ans.get("primary_option")), None)
            if chosen and (not chosen["vetoes"] or not ok):
                pick = chosen
                rationale, concerns, by = ans.get("rationale"), [c for c in ans.get("concerns", []) if isinstance(c, str)][:3], model
            elif ans:
                self.emit("chief", "say", "The model's pick conflicts with a veto; going with the rules.")
        f = next(x for x in finals if x["site"]["id"] == pick["site_id"] and x["time"] == pick["time"])
        if not rationale:
            votes = "; ".join(v["why"] for v in f["votes"].values())
            rationale = f"{pick['name']} at {_fmt_t(pick['time'])} passes {sum(1 for v in f['votes'].values() if v['vote'] == 'GO')} of {len(f['votes'])} checks: {votes}."
        zones = f["analysis"]["zones"]
        backups_regions = [t for t in sorted(table, key=lambda t: (len(t["vetoes"]), -t["points"])) if t["site_id"] != pick["site_id"]]
        return {
            "go": not pick["vetoes"],
            "decided_by": by,
            "primary": {"site_id": pick["site_id"], "name": pick["name"], "landing_time": pick["time"], "zone": zones[0] if zones else None, "vetoes": pick["vetoes"]},
            "backup_zones": [{"id": z["id"], "lat": z["lat"], "lon": z["lon"], "area_km2": z["area_km2"], **z["from_primary"]} for z in zones[1:4]],
            "backup_regions": [{"site_id": t["site_id"], "name": t["name"], "landing_time": t["time"], "vetoes": t["vetoes"]} for t in backups_regions],
            "rationale": rationale,
            "concerns": concerns,
            "table": table,
        }

    async def run(self) -> None:
        try:
            end = self.start + timedelta(days=self.days)
            self.emit("chief", "say", f"Mission brief: choose a landing region and time between {self.start:%d %b} and {end:%d %b} for a {_fmt_h(self.stay_hours)} surface stay. Ada, scout every region.", to="scout")
            top = await self.scout(self.start, self.days)
            if not top:
                self.emit("chief", "say", f"Nothing lands in the requested window. Ada, look at the two weeks after {end:%d %b} so we can offer the earliest slip.", to="scout")
                top = await self.scout(end, 14.0)
            if not top:
                self.emit("chief", "decision", f"NO-GO: no region is landable between {self.start:%d %b} and {end + timedelta(days=14):%d %b}.", decision=None)
                return
            self.emit("chief", "say", f"Finalists: {', '.join(x['site']['name'] for x in top)}. Rocco maps each one; Sol, Echo, Tempo and Nova check them at Rocco's primary zone.")
            finals = [{**x, "time": x["best_time"]} for x in top]
            finals = list(await asyncio.gather(*[self.review(f) for f in finals]))
            finals.sort(key=lambda f: -self.tally(f)[0])
            best = finals[0]
            pts, vetoes = self.tally(best)
            if vetoes and set(vetoes) & {"power", "comms"}:
                names = ", ".join(next(r["name"] for r in ROSTER if r["id"] == a) for a in vetoes)
                self.emit("chief", "say", f"{best['site']['name']} scores well but {names} vetoed it. Tempo, find it another time.", to="timing")
                g = await self.retime(best)
                if g:
                    self.emit("chief", "say", f"{g['site']['name']} works at {_fmt_t(g['time'])}. Adding it to the table.")
                    finals.append(g)
                else:
                    self.emit("timing", "say", f"No clean slot for {best['site']['name']} in this window.", to="chief")
            decision = await self.decide(finals)
            p = decision["primary"]
            if decision["go"]:
                head = f"GO: land at {p['name']}" + (f" zone {p['zone']['id']}" if p["zone"] else "") + f", {_fmt_t(p['landing_time'])}."
            else:
                blockers = ", ".join(next(r["role"] for r in ROSTER if r["id"] == a) for a in p["vetoes"])
                head = f"NO-GO: no option passes every check. Closest is {p['name']} at {_fmt_t(p['landing_time'])}, blocked by {blockers}."
            self.emit("chief", "decision", head + " " + decision["rationale"], decision=decision, headline=head)
        except Exception as e:  # keep the stream well-formed for the UI
            self.emit("chief", "error", f"Mission control error: {e}")
        finally:
            self.q.put_nowait(None)


async def stream(start: datetime, days: float = 14.0, stay_hours: float = 156.0) -> AsyncIterator[str]:
    m = Mission(start, days, stay_hours)
    yield json.dumps({"seq": 0, "t": 0, "agent": "chief", "kind": "roster", "text": "", "roster": ROSTER, "llm": llm_model()}) + "\n"
    task = asyncio.create_task(m.run())
    while True:
        ev = await m.q.get()
        if ev is None:
            break
        yield json.dumps(ev) + "\n"
    await task
