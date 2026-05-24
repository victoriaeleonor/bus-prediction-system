import time
import random
from datetime import datetime
import requests

# bus configuration
BUS_ID = "bus_001"
ROUTE_NAME = "ruta_1"
INTERVAL = 5

# coordinates
ROUTE_COORDINATES = [
    (-34.6037, -58.3816),
    (-34.6045, -58.3825),
    (-34.6053, -58.3834),
    (-34.6061, -58.3843),
    (-34.6069, -58.3852),
]

def get_next_position(index):
    return ROUTE_COORDINATES[index % len(ROUTE_COORDINATES)]

def is_rush_hour(hour):
    return 1 if (7 <= hour <= 9) or (18 <= hour <= 20) else 0

def build_payload(index):
    lat, lon = get_next_position(index)
    now = datetime.now()
    total_stops = len(ROUTE_COORDINATES)
    
    return {
        "bus_id": BUS_ID,
        "route": ROUTE_NAME,
        "lat": lat,
        "lon": lon,
        "timestamp": now.isoformat(),
        "occupancy": random.randint(0, 100),
        "hour": now.hour,
        "day_of_week": now.weekday(),  # 0=lunes, 6=domingo
        "is_rush_hour": is_rush_hour(now.hour),
        "route_progress": round(index % total_stops / total_stops, 2)
    }


BACKEND_URL = "http://localhost:3000/predict/eta" #replace with the real url later

def send_data(payload):
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=5)
        print(f"[OK] Sent: {payload['timestamp']} | Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[SIM] No backend yet — payload: {payload['bus_id']} @ {payload['timestamp']}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    print("Simulator started. Sending every 5 seconds...")
    i = 0
    while True:
        payload = build_payload(i)
        send_data(payload)
        i += 1
        time.sleep(INTERVAL)
