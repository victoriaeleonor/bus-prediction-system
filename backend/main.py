"""
Bus Prediction System — FastAPI Backend
Loads occupancy (XGBoost/RF) and ETA (RF) models and serves predictions.
"""
import os
import json
import math
import pickle
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── paths ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
OCC_DIR  = BASE / "ml" / "occupancy" / "models"
ETA_DIR  = BASE / "ml" / "eta" / "model"

# ── load occupancy models ──────────────────────────────────────────────────
SUFFIX = "with_lags"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

xgb_model      = load_pkl(OCC_DIR / f"xgboost_occupancy_{SUFFIX}.pkl")
xgb_encoder    = load_pkl(OCC_DIR / f"xgboost_label_encoder_{SUFFIX}.pkl")
xgb_features   = load_pkl(OCC_DIR / f"xgboost_feature_names_{SUFFIX}.pkl")
label_encoders = load_pkl(BASE / "ml" / "occupancy" / "data" / f"sunt_2024_03_{SUFFIX}_encoders.pkl")

try:
    eta_model = joblib.load(ETA_DIR / "rf_model_eta.pkl")
    print("✓ ETA model loaded")
except FileNotFoundError:
    eta_model = None
    print("⚠ ETA model not found — will return dummy ETA")

print("✓ All models loaded")

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Bus Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── pydantic schemas ───────────────────────────────────────────────────────
class BusPayload(BaseModel):
    bus_id: str
    route: str
    lat: float
    lon: float
    timestamp: str
    occupancy: int
    hour: int
    day_of_week: int
    is_rush_hour: int
    route_progress: float
    stop_index: int = 0   # ← nuevo, default 0 para compatibilidad

class PredictionResponse(BaseModel):
    bus_id: str
    route: str
    lat: float
    lon: float
    timestamp: str
    occupancy_raw: int
    occupancy_class: str
    occupancy_pct: float
    eta_minutes: float
    hour: int
    day_of_week: int
    is_rush_hour: int

# ── route stops (loaded from route_38.json) ───────────────────────────────
def load_route_from_json() -> list[tuple[float, float]]:
    route_file = BASE / "raspberry-pi" / "route_38.json"
    with open(route_file) as f:
        data = json.load(f)
    node_map = {el["id"]: (el["lat"], el["lon"]) for el in data["elements"] if el["type"] == "node"}
    way_map  = {el["id"]: el["nodes"]            for el in data["elements"] if el["type"] == "way"}
    relation = next(el for el in data["elements"] if el["type"] == "relation")
    route, prev_last = [], None
    for m in relation["members"]:
        if m["type"] != "way" or m["ref"] not in way_map:
            continue
        coords = [node_map[n] for n in way_map[m["ref"]] if n in node_map]
        if not coords:
            continue
        if prev_last:
            d_fwd = abs(coords[0][0]  - prev_last[0]) + abs(coords[0][1]  - prev_last[1])
            d_rev = abs(coords[-1][0] - prev_last[0]) + abs(coords[-1][1] - prev_last[1])
            if d_rev < d_fwd:
                coords = coords[::-1]
            if min(d_fwd, d_rev) > 0.015:
                continue
        route.extend(coords)
        prev_last = route[-1]
    return route[::2]   # every 2nd point keeps the list manageable

# ── route GPS trace (for map drawing only) ────────────────────────────────
ROUTE_COORDS = load_route_from_json()
print(f"✓ Route loaded: {len(ROUTE_COORDS)} points from route_38.json")

# ── actual bus stops (for ETA calculation) ────────────────────────────────
# These are the real Línea 38 stop positions along the route, in order.
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
OCC_LABELS = {0: "low", 1: "medium", 2: "high", 3: "very_high"}

def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def get_next_stop(lat, lon, stop_index):
    """Distance in meters to the next actual bus stop.
    stop_index is the last stop the bus passed; we return distance to stop_index+1.
    """
    next_idx = (stop_index + 1) % len(BUS_STOPS)
    next_stop = BUS_STOPS[next_idx]
    return haversine_meters(lat, lon, next_stop[0], next_stop[1])

def predict_occupancy(payload: BusPayload) -> tuple[str, float]:
    now = datetime.fromisoformat(payload.timestamp)
    row = {f: 0 for f in xgb_features}
    mapping = {
        "hour": now.hour,
        "day_of_week": payload.day_of_week,
        "is_rush_hour": payload.is_rush_hour,
        "route_progress": payload.route_progress,
        "loading": payload.occupancy / 100.0,
        "month": now.month,
    }
    for k, v in mapping.items():
        if k in row:
            row[k] = v
    if "route_short_name" in row and "route_short_name" in label_encoders:
        enc = label_encoders["route_short_name"]
        try:
            row["route_short_name"] = enc.transform([payload.route])[0]
        except Exception:
            row["route_short_name"] = 0
    X = pd.DataFrame([row])[xgb_features]
    pred_encoded = xgb_model.predict(X)[0]
    occ_class = xgb_encoder.inverse_transform([pred_encoded])[0] \
        if hasattr(xgb_encoder, "inverse_transform") else OCC_LABELS.get(pred_encoded, "medium")
    pct_map = {"low": 15.0, "medium": 40.0, "high": 65.0, "very_high": 88.0}
    return str(occ_class), pct_map.get(str(occ_class), 50.0)

def predict_eta(payload: BusPayload) -> float:
    if eta_model is None:
        return round((1 - payload.route_progress) * 5, 2)
    now = datetime.fromisoformat(payload.timestamp)
    distance = get_next_stop(payload.lat, payload.lon, payload.stop_index)
    X = pd.DataFrame([{
        "DistanceFromStop": distance,
        "month":       now.month,
        "day":         now.day,
        "day_of_week": payload.day_of_week,
        "hour":        payload.hour,
        "minute":      now.minute,
    }])
    eta = float(eta_model.predict(X)[0])
    return round(max(0.1, eta), 2)

# ── routes ─────────────────────────────────────────────────────────────────
@app.post("/predict/eta", response_model=PredictionResponse)
async def predict(payload: BusPayload):
    occ_class, occ_pct = predict_occupancy(payload)
    eta = predict_eta(payload)
    return PredictionResponse(
        bus_id          = payload.bus_id,
        route           = payload.route,
        lat             = payload.lat,
        lon             = payload.lon,
        timestamp       = payload.timestamp,
        occupancy_raw   = payload.occupancy,
        occupancy_class = occ_class,
        occupancy_pct   = occ_pct,
        eta_minutes     = eta,
        hour            = payload.hour,
        day_of_week     = payload.day_of_week,
        is_rush_hour    = payload.is_rush_hour,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "models": ["xgboost_occupancy", "rf_eta"]}

@app.get("/route")
async def get_route():
    """Return full route GPS trace and actual bus stop positions for the frontend map."""
    return {
        "coordinates": [list(p) for p in ROUTE_COORDS],
        "stops":       [list(p) for p in BUS_STOPS],
    }

# ── WebSocket broadcast ────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

@app.post("/predict/eta/broadcast", response_model=PredictionResponse)
async def predict_and_broadcast(payload: BusPayload):
    result = await predict(payload)
    print(f"[WS] clientes conectados: {len(manager.active)}")
    await manager.broadcast(result.dict())
    return result

# ── serve frontend ─────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_path / "index.html"))