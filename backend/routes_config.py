"""
routes_config.py
=================
Registry of the bus lines the system knows about.

Adding a real line means adding one entry to LINES below plus its
route/stops/segment-times data files under backend/data/ and
raspberry-pi/ — no other backend module needs to change.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RASPBERRY_DIR = BASE_DIR.parent / "raspberry-pi"

DEFAULT_LINE = "38"


class LineConfig:
    def __init__(self, route_id: str, label: str, route_file: str, stop_names_file: str, segments_file: str):
        self.route_id = route_id
        self.label = label
        self.route_file = RASPBERRY_DIR / route_file

        with open(DATA_DIR / stop_names_file, encoding="utf-8") as f:
            stops = sorted(json.load(f)["stops"], key=lambda s: s["index"])
        self.bus_stops = [(s["lat"], s["lon"]) for s in stops]
        self.stop_names = [s["street"] for s in stops]

        with open(DATA_DIR / segments_file, encoding="utf-8") as f:
            self.segments = json.load(f)["segments"]


LINES = {
    "38": LineConfig(
        route_id="38",
        label="Línea 38",
        route_file="route_38.json",
        stop_names_file="stop_names.json",
        segments_file="segment_times.json",
    ),
    "15-1": LineConfig(
        route_id="15-1",
        label="Línea 15-1",
        route_file="route_15_1.json",
        stop_names_file="stop_names_15_1.json",
        segments_file="segment_times_15_1.json",
    ),
}


def get_line(route_id: str) -> LineConfig:
    """Returns the LineConfig for route_id, falling back to DEFAULT_LINE
    for unknown/missing ids (e.g. an older client that predates route_id)."""
    return LINES.get(route_id, LINES[DEFAULT_LINE])
