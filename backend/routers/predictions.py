"""
routers/predictions.py
======================
Prediction and WebSocket endpoints.

Exposed Endpoints
-----------------
POST /predict/eta
    Receives bus data from the simulator (or from the real Raspberry Pi),
    runs both ML models, and returns occupancy class + ETA in minutes.

POST /predict/eta/broadcast
    Same as /predict/eta, but also broadcasts the result
    to all connected WebSocket clients. This is the endpoint
    currently used by the simulator (BACKEND_URL in simulator.py).

WS /ws
    WebSocket endpoint. The frontend connects here to receive
    real-time updates without polling.

Models
------
- Occupancy: XGBoost trained on the SUNT OD dataset.
  Features: hour, day_of_week, is_rush_hour, route_progress, loading,
            month, route_short_name (encoded), + lags from previous stops.

- ETA: XGBoost/RF trained on the MTA dataset
  Features: DistanceFromStop, dist_to_dest_m, distance_close,
            schedule_delay, speed_kmh, speed_roll3, direction,
            hour_sin, hour_cos, is_am_rush, is_pm_rush, proximity_enc.

  Important: the ETA model was trained on NYC (MTA) data.
  The features are generic (distance, speed, time), so it
  works reasonably well in other contexts.

Model Access
------------
Models are loaded in main.py during startup and stored in app.state.models.
Endpoints access them via request.app.state.models — without globals or
cross-imports from main.py.
"""

import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from services.ws_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


# ── schemas ────────────────────────────────────────────────────────────────
# Same fields as the original main.py — the simulator does not need changes.

class BusPayload(BaseModel):
    bus_id: str
    route: str
    lat: float
    lon: float
    timestamp: str
    occupancy: int   # current passengers (approximately 0–100)
    hour: int
    day_of_week: int
    is_rush_hour: int
    route_progress: float  # 0.0 → 1.0, relative position on the route
    stop_index: int = 0  # index of the last stop passed by the bus
    speed_kmh: float = 0.0  # instantaneous speed calculated by the simulator


class PredictionResponse(BaseModel):
    bus_id: str
    route: str
    lat: float
    lon: float
    timestamp: str
    occupancy_raw: int
    occupancy_class: str  # "low" | "medium" | "high" | "very_high"
    occupancy_pct: float  # percentage representation of occupancy level
    eta_minutes: float  # minutes until the next stop
    hour: int
    day_of_week: int
    is_rush_hour: int


# ── Line 38 bus stops (for ETA calculation) ───────────────────────────────
# Same array that existed in the original main.py. If more routes are added
# in the future, this should be moved to a configuration file or database.

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
]

OCC_LABELS = {0: "low", 1: "medium", 2: "high", 3: "very_high"}
PCT_MAP    = {"low": 15.0, "medium": 40.0, "high": 65.0, "very_high": 88.0}


# ── geometric helpers ─────────────────────────────────────────────────────

def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _dist_to_next_stop(lat: float, lon: float, stop_index: int) -> float:
    """Distance in meters to the next stop (stop_index + 1)."""
    next_idx = (stop_index + 1) % len(BUS_STOPS)
    ns = BUS_STOPS[next_idx]
    return _haversine_meters(lat, lon, ns[0], ns[1])


# ── prediction logic ──────────────────────────────────────────────────────

def _predict_occupancy(payload: BusPayload, models: dict) -> tuple[str, float]:
    """
    Runs the occupancy XGBoost model (trained on SUNT OD).
    Returns (occupancy_class, occupancy_pct).

    The 'loading' feature is the passenger-to-capacity ratio,
    which is what the model saw during training with SUNT OD.
    The simulator sends 'occupancy' as an integer from 0–100,
    so we divide it by 100 to recover the ratio.
    """
    xgb_model      = models["xgb_model"]
    xgb_encoder    = models["xgb_encoder"]
    xgb_features   = models["xgb_features"]
    label_encoders = models["label_encoders"]

    now = datetime.fromisoformat(payload.timestamp)

    row = {f: 0 for f in xgb_features}
    direct_mapping = {
        "hour":           now.hour,
        "day_of_week":    payload.day_of_week,
        "is_rush_hour":   payload.is_rush_hour,
        "route_progress": payload.route_progress,
        "loading":        payload.occupancy / 100.0,
        "month":          now.month,
    }

    for k, v in direct_mapping.items():
        if k in row:
            row[k] = v

    # route_short_name must go through the training LabelEncoder
    if "route_short_name" in row and "route_short_name" in label_encoders:
        enc = label_encoders["route_short_name"]
        try:
            row["route_short_name"] = enc.transform([payload.route])[0]
        except Exception:
            # Unknown route for the encoder → use 0 (most common class)
            row["route_short_name"] = 0

    X = pd.DataFrame([row])[xgb_features]
    pred_encoded = xgb_model.predict(X)[0]

    if hasattr(xgb_encoder, "inverse_transform"):
        occ_class = str(xgb_encoder.inverse_transform([pred_encoded])[0])
    else:
        occ_class = OCC_LABELS.get(int(pred_encoded), "medium")

    return occ_class, PCT_MAP.get(occ_class, 50.0)


def _predict_eta(payload: BusPayload, models: dict) -> float:
    """
    Runs the ETA model (XGBoost/RF trained on MTA data).
    Returns the estimated minutes until the next stop.

    If the model is not loaded (eta_model is None), returns a
    dummy value based on route_progress — same behavior as the
    original main.py.

    Features built here must exactly match the training features.
    The final order is enforced with [eta_features] when creating
    the DataFrame, just like in the original main.py.
    """
    eta_model = models.get("eta_model")
    eta_features = models.get("eta_features", [])

    if eta_model is None:
        return round((1 - payload.route_progress) * 5, 2)

    now = datetime.fromisoformat(payload.timestamp)

    distance = _dist_to_next_stop(payload.lat, payload.lon, payload.stop_index)
    terminal = BUS_STOPS[-1]
    dist_to_dest = _haversine_meters(
        payload.lat,
        payload.lon,
        terminal[0],
        terminal[1],
    )

    speed = max(0.0, min(60.0, payload.speed_kmh))

    # speed_roll3: average with the previous speed if available.
    # The simulator does not send previous speed, so we use speed directly
    # (same approach as the original main.py with prev_speed=None).
    speed_roll3 = speed

    hour_sin = math.sin(2 * math.pi * now.hour / 24)
    hour_cos = math.cos(2 * math.pi * now.hour / 24)

    row = {
        "DistanceFromStop": distance,
        "dist_to_dest_m":   dist_to_dest,
        "distance_close":   int(distance < 300),
        "schedule_delay":   0.0,
        "speed_kmh":        speed,
        "speed_roll3":      speed_roll3,
        "direction":        1,
        "hour_sin":         hour_sin,
        "hour_cos":         hour_cos,
        "is_am_rush":       int(now.hour in (7, 8, 9)),
        "is_pm_rush":       int(now.hour in (16, 17, 18)),
        "proximity_enc":    1 if distance > 300 else 0,
    }

    # Reorder columns exactly as in training
    X = pd.DataFrame([row])[eta_features]
    eta = float(eta_model.predict(X)[0])
    return round(max(0.1, eta), 2)


# ── endpoints ─────────────────────────────────────────────────────────────

@router.post(
    "/predict/eta",
    response_model=PredictionResponse,
    summary="Predict ETA and occupancy level",
)
async def predict(payload: BusPayload, request: Request):
    """
    Receives current bus data and returns:
    - occupancy_class: occupancy level predicted by XGBoost
    - eta_minutes: minutes until the next stop predicted by XGBoost/RF

    Used by the simulator and (in the future) by the real Raspberry Pi.
    """