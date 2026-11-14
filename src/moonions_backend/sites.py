"""Curated candidate sites.

Historic landing coordinates are published values. Artemis III region centres are
APPROXIMATE (NASA publishes 15x15 km regions, not points); use the /api/suggest search to
find the best spot inside a region.
"""

from __future__ import annotations

SITES: list[dict] = [
    # --- Artemis III candidate regions (NASA, Oct 2024), approximate centres ---
    {"id": "cabeus-b-peak", "name": "Peak near Cabeus B", "lat": -84.6, "lon": -46.0, "group": "Artemis III region", "approx": True},
    {"id": "haworth", "name": "Haworth", "lat": -86.9, "lon": -9.0, "group": "Artemis III region", "approx": True},
    {"id": "malapert", "name": "Malapert Massif", "lat": -85.95, "lon": 2.5, "group": "Artemis III region", "approx": True},
    {"id": "mouton-plateau", "name": "Mons Mouton Plateau", "lat": -84.7, "lon": 31.0, "group": "Artemis III region", "approx": True},
    {"id": "mouton", "name": "Mons Mouton", "lat": -85.3, "lon": 35.5, "group": "Artemis III region", "approx": True},
    {"id": "nobile-rim-1", "name": "Nobile Rim 1", "lat": -84.3, "lon": 60.0, "group": "Artemis III region", "approx": True},
    {"id": "nobile-rim-2", "name": "Nobile Rim 2", "lat": -83.9, "lon": 66.0, "group": "Artemis III region", "approx": True},
    {"id": "de-gerlache-rim-2", "name": "de Gerlache Rim 2", "lat": -88.6, "lon": -66.0, "group": "Artemis III region", "approx": True},
    {"id": "slater-plain", "name": "Slater Plain", "lat": -87.9, "lon": 113.0, "group": "Artemis III region", "approx": True},
    {"id": "shackleton-ridge", "name": "Shackleton connecting ridge", "lat": -89.45, "lon": -137.0, "group": "South pole", "approx": True},
    # --- Historic landings (reference points; known-good terrain) ---
    {"id": "apollo-11", "name": "Apollo 11 · Tranquility Base", "lat": 0.67408, "lon": 23.47297, "group": "Historic landing"},
    {"id": "apollo-12", "name": "Apollo 12", "lat": -3.01239, "lon": -23.42157, "group": "Historic landing"},
    {"id": "apollo-14", "name": "Apollo 14", "lat": -3.6453, "lon": -17.47136, "group": "Historic landing"},
    {"id": "apollo-15", "name": "Apollo 15 · Hadley Rille", "lat": 26.13222, "lon": 3.63386, "group": "Historic landing"},
    {"id": "apollo-16", "name": "Apollo 16 · Descartes", "lat": -8.97301, "lon": 15.50019, "group": "Historic landing"},
    {"id": "apollo-17", "name": "Apollo 17 · Taurus-Littrow", "lat": 20.1908, "lon": 30.77168, "group": "Historic landing"},
    {"id": "change-4", "name": "Chang'e 4 · Von Kármán", "lat": -45.4446, "lon": 177.5991, "group": "Historic landing"},
    {"id": "change-6", "name": "Chang'e 6 · Apollo basin", "lat": -41.64, "lon": -153.99, "group": "Historic landing"},
    {"id": "chandrayaan-3", "name": "Chandrayaan-3 · Shiv Shakti", "lat": -69.373, "lon": 32.319, "group": "Historic landing"},
    {"id": "slim", "name": "SLIM · Shioli", "lat": -13.316, "lon": 25.251, "group": "Historic landing"},
    {"id": "im-1", "name": "IM-1 Odysseus · Malapert A", "lat": -80.13, "lon": 1.44, "group": "Historic landing"},
    {"id": "blue-ghost-1", "name": "Blue Ghost M1 · Mare Crisium", "lat": 18.56, "lon": 61.81, "group": "Historic landing"},
]


def get_site(site_id: str) -> dict | None:
    return next((s for s in SITES if s["id"] == site_id), None)
