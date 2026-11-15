"""Bake the optimal touchdown time and landing zones of every candidate region into files.

For each candidate region (historic landing sites excluded):
  1. coarse: the region landing timeline over the next 32 days (6-hourly) -> best score
     (score = sqrt(viable share of the region) x best landing spot, as in the dashboard)
  2. fine: the hourly region analysis over the 24 hours around it -> best hour
  3. that hour's landing zones (Z1 primary + backups), region stats, and Sun / Earth / surface
     temperature at the region centre over the surface stay

Writes exports/optimal_zones.json (everything) and exports/optimal_zones.csv (one row per zone).
Uses the running backend's API (default http://localhost:8000), so all its caches are reused.

    uv run python scripts/bake_optimal.py [--api URL] [--start ISO] [--days 32] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


def get(api: str, path: str, **params) -> dict:
    url = f"{api}{path}?{urllib.parse.urlencode(params)}" if params else f"{api}{path}"
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.loads(r.read())


def score(stats: dict) -> float:
    return math.sqrt(max(0.0, stats["viable_share"])) * stats["best_success"]


def main(api: str, start: datetime, days: int, out: Path) -> None:
    sites = [s for s in get(api, "/api/sites") if s.get("group") != "Historic landing"]
    results = []
    for s in sites:
        half = s.get("half_size_m") or 7500.0
        tl = get(api, "/api/regions/timeline", lat=s["lat"], lon=s["lon"], start=start.isoformat(), half_size_m=half, days=days, step_hours=6)
        if not tl.get("available"):
            continue
        i = max(range(len(tl["score"])), key=lambda k: tl["score"][k])
        coarse = datetime.fromisoformat(tl["times"][i])
        lo = max(start, coarse - timedelta(hours=12))
        rng = get(api, "/api/regions/analyze-range", lat=s["lat"], lon=s["lon"], start=lo.isoformat(), hours=24, half_size_m=half)
        best = max(rng["frames"], key=lambda f: score(f["stats"]))
        t = best["landing_time"]
        cond = get(api, "/api/conditions", lat=s["lat"], lon=s["lon"], start=t, days=14)
        results.append(
            {
                "id": s["id"],
                "name": s["name"],
                "group": s["group"],
                "centre": {"lat": s["lat"], "lon": s["lon"], "approximate": bool(s.get("approx"))},
                "region_size_km": 2 * half / 1000,
                "optimal_touchdown_utc": t,
                "score": round(score(best["stats"]), 4),
                "region": best["stats"],
                "zones": best["zones"],
                "centre_conditions": {
                    "sun_up_at_touchdown": cond["now"]["sun_up"],
                    "earth_in_view_at_touchdown": cond["now"]["earth_up"],
                    "sun_elevation_deg": cond["now"]["sun_elevation_deg"],
                    "next_sunset": cond["next"]["sunset"],
                    "stay_hours": cond["stay"]["hours"],
                    "stay_power_ok": cond["stay"]["power_ok"],
                    "stay_lowest_battery": cond["stay"]["lowest_battery"],
                    "stay_sunlit_share": cond["stay"]["sunlit_share"],
                    "stay_earth_share": cond["stay"]["earth_share"],
                    "surface_temp_k": cond.get("thermal"),
                },
                "landing_windows": tl["windows"],
            }
        )
        print(f"{s['name']:32s} {t}  viable {best['stats']['viable_share']:.0%}  zones {len(best['zones'])}", flush=True)
    results.sort(key=lambda r: -r["score"])
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_window": {"start": start.isoformat(), "days": days},
        "method": "Best hour by sqrt(viable share) x best landing spot: 6-hourly region timeline, refined hourly over 24 h. "
        "Zones = connected areas where a landing at that hour is predicted to succeed (AI touchdown x sunlight, power and "
        "Earth link) that fit the 3-sigma landing ellipse; Z1 is the primary target, the rest are backups.",
        "regions": results,
    }
    (out / "optimal_zones.json").write_text(json.dumps(doc, indent=1))
    with open(out / "optimal_zones.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "optimal_touchdown_utc", "region_viable_share", "zone", "rank", "lat", "lon", "area_km2", "mean_success", "clearance_m", "from_primary_m", "direction", "decide_by_s"])
        for r in results:
            for z in r["zones"]:
                fp = z.get("from_primary") or {}
                w.writerow([r["name"], r["optimal_touchdown_utc"], r["region"]["viable_share"], z["id"], z.get("rank"), round(z["lat"], 5), round(z["lon"], 5), z.get("area_km2"), z.get("mean_success"), z.get("margin_m"), fp.get("distance_m", ""), fp.get("direction", ""), fp.get("decide_by_s", "")])
    print("wrote", out / "optimal_zones.json", "and", out / "optimal_zones.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--start", default=None, help="ISO start (default: today 00:00 UTC)")
    ap.add_argument("--days", type=int, default=32)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "exports"))
    a = ap.parse_args()
    st = datetime.fromisoformat(a.start) if a.start else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    main(a.api, st, a.days, Path(a.out))
