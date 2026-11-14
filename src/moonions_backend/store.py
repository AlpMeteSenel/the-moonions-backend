"""SQLite store for landing proposals (auditable decisions)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    landing_time TEXT NOT NULL,
    profile TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    overall REAL,
    verdict TEXT,
    notes TEXT DEFAULT '',
    decided_by TEXT,
    snapshot TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    decided_at TEXT
);
"""

STATUSES = {"pending", "accepted", "rejected"}


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(r: sqlite3.Row, full: bool = True) -> dict:
    d = dict(r)
    snap = json.loads(d.pop("snapshot"))
    if full:
        d["evaluation"] = snap
    else:
        d["backups"] = len(snap.get("backups") or [])
    return d


def create(name: str, evaluation: dict, notes: str = "") -> dict:
    pid = uuid.uuid4().hex[:10]
    now = _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO proposals (id,name,lat,lon,landing_time,profile,overall,verdict,notes,snapshot,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, name, evaluation["lat"], evaluation["lon"], evaluation["landing_time"], evaluation["profile"], evaluation["overall"], evaluation["verdict"], notes, json.dumps(evaluation), now, now),
        )
    return get(pid)


def list_all() -> list[dict]:
    with _conn() as c:
        return [_row(r, full=False) for r in c.execute("SELECT * FROM proposals ORDER BY created_at DESC")]


def get(pid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
        return _row(r) if r else None


def update(pid: str, status: str | None = None, notes: str | None = None, decided_by: str | None = None, name: str | None = None) -> dict | None:
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    sets, args = ["updated_at=?"], [_now()]
    if status is not None:
        sets += ["status=?", "decided_at=?"]
        args += [status, _now() if status != "pending" else None]
    for col, val in (("notes", notes), ("decided_by", decided_by), ("name", name)):
        if val is not None:
            sets.append(f"{col}=?")
            args.append(val)
    with _conn() as c:
        c.execute(f"UPDATE proposals SET {', '.join(sets)} WHERE id=?", (*args, pid))
    return get(pid)


def delete(pid: str) -> bool:
    with _conn() as c:
        return c.execute("DELETE FROM proposals WHERE id=?", (pid,)).rowcount > 0
