import time
import random
import json
import requests
from datetime import datetime
from pathlib import Path

# ── configuración ──────────────────────────────────────────────────────────
BUS_ID     = "bus_001"
ROUTE_NAME = "Línea 38"
INTERVAL   = 5        # segundos entre actualizaciones
MAX_STOPS  = 100      # máximo de puntos a usar de la ruta

# URL del backend — cambiar IP si el simulador corre en la Raspberry Pi
# Mismo equipo:     http://localhost:8000/predict/eta/broadcast
# Raspberry Pi:     http://<IP-de-tu-Mac>:8000/predict/eta/broadcast
BACKEND_URL = "http://localhost:8000/predict/eta/broadcast"

# ── carga de ruta ──────────────────────────────────────────────────────────

def fetch_route_from_file():
    route_file = Path(__file__).parent / "route_38.json"
    with open(route_file) as f:
        data = json.load(f)

    # construir mapa de nodos
    node_map = {el['id']: (el['lat'], el['lon'])
                for el in data['elements'] if el['type'] == 'node'}
    way_map  = {el['id']: el['nodes']
                for el in data['elements'] if el['type'] == 'way'}

    # si no hay relation/ways, el archivo es solo nodos — usar fallback
    ways = [el for el in data['elements'] if el['type'] == 'way']
    relation = next((el for el in data['elements'] if el['type'] == 'relation'), None)

    if not relation or not ways:
        raise ValueError("archivo sin relacion/ways — usar fallback")

    # reconstruir ruta en orden siguiendo los ways de la relacion
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
            d_fwd = abs(coords[0][0]-prev_last[0]) + abs(coords[0][1]-prev_last[1])
            d_rev = abs(coords[-1][0]-prev_last[0]) + abs(coords[-1][1]-prev_last[1])
            if d_rev < d_fwd:
                coords = coords[::-1]
            if min(d_fwd, d_rev) > 0.015:
                continue  # segmento discontinuo — saltear
        route.extend(coords)
        prev_last = route[-1]

    if len(route) < 10:
        raise ValueError("ruta muy corta")

    # usar solo cada 2do punto
    route = route[::2]
    print(f"✓ {len(route)} puntos cargados en orden correcto desde route_38.json")
    return route


def fetch_route_from_osm(relation_id=3984378):
    """
    Descarga la ruta directamente desde Overpass API.
    Funciona si hay conexión a internet sin restricciones SSL.
    """
    print("Descargando ruta desde OpenStreetMap...")
    query = f"[out:json];relation({relation_id});way(r);node(w);out skel qt;"

    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    for server in servers:
        try:
            print(f"  intentando {server}...")
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
                print(f"✓ {len(coords)} puntos descargados desde OSM")
                return coords
        except Exception as e:
            print(f"  falló: {e}")

    raise ValueError("todos los servidores fallaron")


def load_route():
    """Load route from local route_38.json, fall back to Overpass API if missing."""
    try:
        return fetch_route_from_file()
    except FileNotFoundError:
        print("route_38.json not found, trying OSM online...")
    except Exception as e:
        print(f"Error reading route_38.json: {e}, trying OSM online...")

    return fetch_route_from_osm()




# cargar ruta al arrancar
ROUTE_COORDINATES = load_route()
print(f"Ruta lista: {len(ROUTE_COORDINATES)} puntos GPS\n")

# actual bus stops for ETA — must match BUS_STOPS in backend/main.py
BUS_STOPS = [
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
    (-25.3078456, -57.552419 ),
    (-25.3034737, -57.5603186),
    (-25.2977134, -57.5714751),
    (-25.2944599, -57.5789756),
    (-25.2900301, -57.5890618),
    (-25.2839356, -57.5878745),
    (-25.2707073, -57.5838988),
    (-25.2587977, -57.5798663),
    (-25.2537729, -57.5749772),
    (-25.2525967, -57.5772245),
]

# ── helpers ────────────────────────────────────────────────────────────────

def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6_371_000
    import math
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((phi2-phi1)/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(math.radians((lon2-lon1)/2))**2
    return R * 2 * math.asin(math.sqrt(a))

# ── stop tracker (stateful) ────────────────────────────────────────────────
# Advances only when the bus comes within ARRIVAL_THRESHOLD of the next stop.
# This ensures ETA always decreases as the bus approaches, never jumps back up.
ARRIVAL_THRESHOLD = 200  # meters — bus is considered "at" a stop within this distance
_current_stop_idx = 0    # last stop the bus passed; backend adds 1 for "next stop"

def update_stop_index(lat, lon):
    """Advance _current_stop_idx if the bus has arrived at the next stop."""
    global _current_stop_idx
    next_idx = (_current_stop_idx + 1) % len(BUS_STOPS)
    next_stop = BUS_STOPS[next_idx]
    dist = haversine_meters(lat, lon, next_stop[0], next_stop[1])
    if dist <= ARRIVAL_THRESHOLD:
        _current_stop_idx = next_idx
    return _current_stop_idx

def get_next_position(index):
    total = len(ROUTE_COORDINATES)
    cycle = index % (total * 2)
    if cycle < total:
        return ROUTE_COORDINATES[cycle]           # outbound
    else:
        return ROUTE_COORDINATES[total * 2 - 1 - cycle]  # return


def is_rush_hour(hour):
    return 1 if (7 <= hour <= 9) or (18 <= hour <= 20) else 0


def get_occupancy(hour):
    """Simula ocupación realista según hora del día."""
    if 7 <= hour <= 9 or 18 <= hour <= 20:   # hora pico
        return random.randint(60, 100)
    elif 12 <= hour <= 14:                     # mediodía
        return random.randint(30, 60)
    elif 22 <= hour or hour <= 5:             # madrugada
        return random.randint(0, 20)
    else:                                      # resto del día
        return random.randint(10, 40)


def build_payload(index):
    lat, lon = get_next_position(index)
    now = datetime.now()
    total_gps = len(ROUTE_COORDINATES)
    current_gps_idx = index % total_gps

    return {
        "bus_id":         BUS_ID,
        "route":          ROUTE_NAME,
        "lat":            lat,
        "lon":            lon,
        "timestamp":      now.isoformat(),
        "occupancy":      get_occupancy(now.hour),
        "hour":           now.hour,
        "day_of_week":    now.weekday(),
        "is_rush_hour":   is_rush_hour(now.hour),
        "route_progress": round(current_gps_idx / total_gps, 2),
        "stop_index":     update_stop_index(lat, lon),  # last passed stop index
    }

# ── envío al backend ───────────────────────────────────────────────────────

def send_data(payload):
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=5)
        data = response.json()
        print(
            f"[{payload['timestamp'][11:19]}] "
            f"{data.get('bus_id')} | "
            f"occ: {data.get('occupancy_class'):<9} "
            f"({data.get('occupancy_pct', 0):.0f}%) | "
            f"ETA: {data.get('eta_minutes', 0):.1f} min"
        )
    except requests.exceptions.ConnectionError:
        print("[NO BACKEND] ¿Está corriendo uvicorn backend.main:app?")
    except Exception as e:
        print(f"[ERROR] {e}")

# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Simulator iniciado — enviando cada {INTERVAL}s a {BACKEND_URL}")
    print("Dashboard: http://localhost:8000\n")
    i = 0
    while True:
        payload = build_payload(i)
        send_data(payload)
        i += 1
        time.sleep(INTERVAL)