import argparse
import sys
import time
import random
import json
import requests
from datetime import datetime
from pathlib import Path

# Windows consoles often default to a non-UTF-8 codepage (cp1252), which
# can't encode characters like '✓' used in the print statements below —
# without this, that print raises UnicodeEncodeError, which then gets
# swallowed by load_route()'s broad except and misread as a file-read error.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── line registry ────────────────────────────────────────────────────────
# Each real line this simulator can drive. Mirrors backend/routes_config.py
# (kept as a separate, lightweight copy here since this script runs
# standalone, without importing the backend package).
LINES = {
    "38": {
        "bus_id": "bus_001",
        "route_name": "Línea 38",
        "route_id": "38",
        "route_file": "route_38.json",
        "osm_relation_id": 3984378,
        "bus_stops": [
            (-25.3863252, -57.4976859),
            (-25.3786856, -57.4930827),
            (-25.3694164, -57.4916613),
            (-25.3584819, -57.4908329),
            (-25.348588,  -57.5029725),
            (-25.3370013, -57.5099294),
            (-25.3314274, -57.5154413),
            (-25.3193744, -57.5243842),
            (-25.3108736, -57.5307552),
            (-25.304031,  -57.5378214),
            (-25.3062623, -57.5457233),
            (-25.3078456, -57.552419),
            (-25.3034737, -57.5603186),
            (-25.2977134, -57.5714751),
            (-25.2944599, -57.5789756),
            (-25.2900301, -57.5890618),
            (-25.2839356, -57.5878745),
            (-25.2707073, -57.5838988),
            (-25.2587977, -57.5798663),
            (-25.2537729, -57.5749772),
            (-25.2525967, -57.5772245),
        ],
    },
    "15-1": {
        "bus_id": "bus_002",
        "route_name": "Línea 15-1",
        "route_id": "15-1",
        "route_file": "route_15_1.json",
        "osm_relation_id": 3983243,
        "bus_stops": [
            (-25.3770103, -57.5846782),
            (-25.3802046, -57.5923358),
            (-25.3840837, -57.600789),
            (-25.3809364, -57.6075223),
            (-25.374072,  -57.6053603),
            (-25.3670436, -57.6012335),
            (-25.3601039, -57.5971486),
            (-25.3531688, -57.5930268),
            (-25.3465087, -57.5891773),
            (-25.3395993, -57.5850491),
            (-25.3342156, -57.5783753),
            (-25.3315521, -57.573606),
            (-25.3263976, -57.5683834),
            (-25.3213644, -57.561914),
            (-25.3155964, -57.559501),
            (-25.3164865, -57.5657134),
            (-25.3178783, -57.5724406),
            (-25.3253631, -57.5747725),
            (-25.3323874, -57.5762037),
            (-25.337274,  -57.5824752),
            (-25.3434146, -57.5872986),
        ],
    },
}

parser = argparse.ArgumentParser(description="Bus GPS simulator")
parser.add_argument("--line", choices=list(LINES), default="38", help="which line to simulate")
args = parser.parse_args()
CFG = LINES[args.line]

# ── configuration ──────────────────────────────────────────────────────────
BUS_ID     = CFG["bus_id"]
ROUTE_NAME = CFG["route_name"]
ROUTE_ID   = CFG["route_id"]
BUS_STOPS  = CFG["bus_stops"]
INTERVAL   = 5        # seconds between updates
MAX_STOPS  = 100      # maximum number of route points to use (OSM fallback only)

# Backend URL — change the IP if the simulator runs on the Raspberry Pi
# Same machine:    http://localhost:8000/predict/eta/broadcast
# Raspberry Pi:    http://<YOUR-MAC-IP>:8000/predict/eta/broadcast
#                  "http://192.168.X.X:8000/predict/eta/broadcast"
BACKEND_URL = "http://localhost:8000/predict/eta/broadcast"

# ── route loading ──────────────────────────────────────────────────────────

def fetch_route_from_file():
    route_file = Path(__file__).parent / CFG["route_file"]
    with open(route_file) as f:
        data = json.load(f)

    # Build node map
    node_map = {el['id']: (el['lat'], el['lon'])
                for el in data['elements'] if el['type'] == 'node'}
    way_map  = {el['id']: el['nodes']
                for el in data['elements'] if el['type'] == 'way'}

    # If there is no relation/ways, the file contains only nodes — use fallback
    ways = [el for el in data['elements'] if el['type'] == 'way']
    relation = next((el for el in data['elements'] if el['type'] == 'relation'), None)

    if not relation or not ways:
        raise ValueError("file has no relation/ways — use fallback")

    # Rebuild route in order by following the relation ways
    way_members = [m['ref'] for m in relation['members'] if m['type'] == 'way']
    route = []
    prev_last = None

    for way_ref in way_members:
        if way_ref not in way_map:
            continue

        coords = [node_map[n] for n in way_map[way_ref] if n in node_map]

        if not coords:
            continue

        if prev_last:
            d_fwd = abs(coords[0][0] - prev_last[0]) + abs(coords[0][1] - prev_last[1])
            d_rev = abs(coords[-1][0] - prev_last[0]) + abs(coords[-1][1] - prev_last[1])

            if d_rev < d_fwd:
                coords = coords[::-1]

            if min(d_fwd, d_rev) > 0.015:
                continue  # discontinuous segment — skip

        route.extend(coords)
        prev_last = route[-1]

    if len(route) < 10:
        raise ValueError("route is too short")

    # Use every second point only
    route = route[::2]

    print(f"✓ {len(route)} points loaded in correct order from {CFG['route_file']}")
    return route


def fetch_route_from_osm(relation_id=None):
    """
    Download the route directly from the Overpass API.
    Works if internet access is available and SSL restrictions do not apply.
    """
    relation_id = relation_id or CFG["osm_relation_id"]
    print("Downloading route from OpenStreetMap...")
    query = f"[out:json];relation({relation_id});way(r);node(w);out skel qt;"

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    for server in servers:
        try:
            print(f"  trying {server}...")

            r = requests.post(
                server,
                data={"data": query},
                timeout=30,
                verify=False,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if r.status_code == 200 and r.text.strip().startswith("{"):
                data = r.json()

                coords = []
                for el in data["elements"]:
                    if el["type"] == "node":
                        coords.append((el["lat"], el["lon"]))

                coords = sorted(set(coords), key=lambda x: x[1], reverse=True)
                coords = coords[:MAX_STOPS]

                print(f"✓ {len(coords)} points downloaded from OSM")
                return coords

        except Exception as e:
            print(f"  failed: {e}")

    raise ValueError("all servers failed")


def load_route():
    """Load route from the line's local route_*.json, fall back to Overpass API if missing."""
    try:
        return fetch_route_from_file()

    except FileNotFoundError:
        print(f"{CFG['route_file']} not found, trying OSM online...")

    except Exception as e:
        print(f"Error reading {CFG['route_file']}: {e}, trying OSM online...")

    return fetch_route_from_osm()


# Load route at startup
ROUTE_COORDINATES = load_route()
print(f"Route ready: {len(ROUTE_COORDINATES)} GPS points\n")

# ── helpers ────────────────────────────────────────────────────────────────

def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6_371_000
    import math

    phi1, phi2 = math.radians(lat1), math.radians(lat2)

    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2)
        * math.sin(math.radians((lon2 - lon1) / 2)) ** 2
    )

    return R * 2 * math.asin(math.sqrt(a))


# ── speed tracker (stateful) ───────────────────────────────────────────────
_prev_lat: float = None
_prev_lon: float = None
_prev_time: float = None
_smoothed_speed: float = 0.0


def compute_speed_kmh(lat, lon):
    """Compute smoothed speed in km/h from the last known position.

    The route's GPS points (from OSM) aren't evenly spaced — curves have many
    points close together, straight stretches have few far apart — so the
    raw distance/time between two consecutive points swings wildly from tick
    to tick. Exponentially smoothing against the previous reading damps
    those artifacts into something a real bus's speed could plausibly do.
    """
    global _prev_lat, _prev_lon, _prev_time, _smoothed_speed

    now = time.time()
    raw_speed = 0.0

    if _prev_lat is not None:
        dist_m = haversine_meters(_prev_lat, _prev_lon, lat, lon)
        elapsed = now - _prev_time

        if elapsed > 0:
            raw_speed = min(60.0, max(0.0, (dist_m / elapsed) * 3.6))

    _prev_lat, _prev_lon, _prev_time = lat, lon, now
    _smoothed_speed = 0.6 * _smoothed_speed + 0.4 * raw_speed
    speed = _smoothed_speed

    return round(speed, 2)


# Advances (or, on the return leg, retreats) only when the bus comes within
# ARRIVAL_THRESHOLD of the next stop in its current direction of travel.
# This ensures ETA always decreases as the bus approaches and never jumps
# back up — in EITHER direction, not just on the outbound leg.
ARRIVAL_THRESHOLD = 5  # meters — bus is considered at a stop within this distance
_current_stop_idx = 0  # last stop passed; backend adds `travel_direction` for "next stop"


def update_stop_index(lat, lon, direction):
    """Advance/retreat _current_stop_idx if the bus has arrived at the next
    stop in `direction` (+1 outbound, -1 on the return leg)."""
    global _current_stop_idx

    next_idx = (_current_stop_idx + direction) % len(BUS_STOPS)
    next_stop = BUS_STOPS[next_idx]

    dist = haversine_meters(lat, lon, next_stop[0], next_stop[1])

    if dist <= ARRIVAL_THRESHOLD:
        _current_stop_idx = next_idx

    return _current_stop_idx


def get_leg_state(index):
    """Returns ((lat, lon), direction, leg_progress) for `index`.

    direction is +1 while outbound (start -> end of the line) and -1 on the
    return leg (end -> start), so callers always know which way "next stop"
    points. leg_progress is 0->1 within whichever leg is currently active
    (previously route_progress just cycled 0->1 forever regardless of
    direction, which made it meaningless during the return leg).
    """
    total = len(ROUTE_COORDINATES)
    cycle = index % (total * 2)

    if cycle < total:
        return ROUTE_COORDINATES[cycle], 1, cycle / total  # outbound
    else:
        pos_idx = total * 2 - 1 - cycle
        return ROUTE_COORDINATES[pos_idx], -1, (cycle - total) / total  # return trip


def is_rush_hour(hour):
    return 1 if (7 <= hour <= 9) or (18 <= hour <= 20) else 0


def get_occupancy(hour):
    """Simulate realistic occupancy based on time of day."""
    if 7 <= hour <= 9 or 18 <= hour <= 20:  # rush hour
        return random.randint(60, 100)
    elif 12 <= hour <= 14:  # midday
        return random.randint(30, 60)
    elif 22 <= hour or hour <= 5:  # overnight
        return random.randint(0, 20)
    else:  # rest of the day
        return random.randint(10, 40)


def build_payload(index):
    (lat, lon), direction, leg_progress = get_leg_state(index)

    now = datetime.now()

    return {
        "bus_id": BUS_ID,
        "route": ROUTE_NAME,
        "route_id": ROUTE_ID,
        "lat": lat,
        "lon": lon,
        "timestamp": now.isoformat(),
        "occupancy": get_occupancy(now.hour),
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "is_rush_hour": is_rush_hour(now.hour),
        "route_progress": round(leg_progress, 2),
        "stop_index": update_stop_index(lat, lon, direction),
        "travel_direction": direction,
        "speed_kmh": compute_speed_kmh(lat, lon),
    }


# ── send data to backend ───────────────────────────────────────────────────

def send_data(payload):
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=5)
        data = response.json()

        print(
            f"[{payload['timestamp'][11:19]}] "
            f"{data.get('bus_id')} | "
            f"occ: {str(data.get('occupancy_class', '?')):<9} "
            f"({data.get('occupancy_pct') or 0:.0f}%) | "
            f"ETA: {data.get('eta_minutes') or 0:.1f} min"
        )
        
    except requests.exceptions.ConnectionError:
        print("[NO BACKEND] Is uvicorn backend.main:app running?")

    except Exception as e:
        print(f"[ERROR] {e}")


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Simulating {ROUTE_NAME} ({BUS_ID}) — sending every {INTERVAL}s to {BACKEND_URL}")
    print("Dashboard: http://localhost:8000\n")

    i = 0

    while True:
        payload = build_payload(i)
        send_data(payload)
        i += 1
        time.sleep(INTERVAL)
