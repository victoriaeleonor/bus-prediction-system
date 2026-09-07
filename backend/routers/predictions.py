"""
routers/predictions.py
======================
Prediction and WebSocket endpoints.

Exposed Endpoints
-----------------
POST /predict/eta
    Receives bus data from the simulator (or from the actual Raspberry Pi),
    runs both ML models, and returns occupancy class + ETA in minutes.

POST /predict/eta/broadcast
    Same as /predict/eta but also broadcasts the result
    to all connected WebSocket clients. This is the endpoint
    currently used by the simulator (BACKEND_URL in simulator.py).

WS /ws
    WebSocket endpoint. The frontend connects here to receive
    real-time updates without polling.

Models
------
- Occupancy: XGBoost trained with SUNT OD (Salvador, Brazil).
  Features: route_short_name (encoded), direction_id, pt_sequence, stop_id,
            hour, day_of_week, is_weekend, is_rush_hour, route_progression,
            loading_lag_1, loading_lag_2, trip_stage, time_of_day,
            loading_mean_route.

  Known limitations: the live BusPayload has no direction (inbound/outbound)
  signal and no real GTFS pt_sequence/stop_id, so direction_id and stop_id
  are held at fixed defaults and pt_sequence is approximated with
  payload.stop_index. Combined importance of these three is under 2%.
  loading_lag_1/loading_lag_2 (93.6% combined importance) are tracked live
  per bus via an in-memory history — see _predict_occupancy().

- ETA: XGBoost/RF trained with MTA (New York).
  Features: DistanceFromStop, dist_to_dest_m, distance_close,
            schedule_delay, speed_kmh, speed_roll3, direction,
            hour_sin, hour_cos, is_am_rush, is_pm_rush, proximity_enc.

  Important: the ETA model was trained on NYC data (MTA).
  The features are generic (distance, speed, time), which is why
  it works reasonably well in other contexts, but absolute times
  may differ from Paraguayan reality.

Model Access
------------
Models are loaded in main.py at startup and stored in app.state.models.
Endpoints read them via request.app.state.models — without globals or
cross-imports with main.py.
"""

import json
import logging
import math
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

#from services.ws_manager import manager
from backend.services.ws_manager import manager

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
    occupancy: int          # current passengers (approx. 0-100)
    hour: int
    day_of_week: int
    is_rush_hour: int
    route_progress: float   # 0.0 → 1.0, relative position on the route
    stop_index: int = 0     # index of the last stop the bus passed
    speed_kmh: float = 0.0  # instantaneous speed calculated by the simulator


class PredictionResponse(BaseModel):
    bus_id: str
    route: str
    lat: float
    lon: float
    timestamp: str
    occupancy_raw: int
    occupancy_class: str    # "low" | "medium" | "high" | "very_high"
    occupancy_pct: float    # percentage representation of the level
    eta_minutes: float      # minutes until the next stop
    hour: int
    day_of_week: int
    is_rush_hour: int


class TripRequest(BaseModel):
    origin_stop_id: int       # index into BUS_STOPS, 0-based
    destination_stop_id: int  # index into BUS_STOPS, 0-based, must be > origin_stop_id


class TripResponse(BaseModel):
    eta_to_origin_min: float   # minutes for the bus to reach origin_stop_id
    trip_duration_min: float   # minutes from origin_stop_id to destination_stop_id
    total_arrival_time: str    # ISO timestamp: now + eta_to_origin + trip_duration


# ── Line 38 stops (for ETA calculation) ──────────────────────────
# Same array that was in main.py. If more lines are added in the future,
# this would be moved to a configuration file or database.

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

OCC_LABELS = {0: "low", 1: "medium", 2: "high", 3: "very_high"}
PCT_MAP    = {"low": 15.0, "medium": 40.0, "high": 65.0, "very_high": 88.0}


# ── segment time table (trip planner) ────────────────────────────────────
# Static table of average travel time between consecutive stops, derived
# once (offline, see scripts) from the 548-point simulated GPS route
# (raspberry-pi/route_38.json, one point every INTERVAL=5s in simulator.py).
# Loaded once at import time — never recalculated per request.

_SEGMENTS_PATH = Path(__file__).parent.parent / "data" / "segment_times.json"
with open(_SEGMENTS_PATH) as _f:
    _SEGMENT_DATA = json.load(_f)
SEGMENTS = _SEGMENT_DATA["segments"]  # list of 20 {from_stop, to_stop, avg_time_min, distance_m, avg_speed_kmh}


# ── geometric helpers ────────────────────────────────────────────────────

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
    next_stop = BUS_STOPS[next_idx]
    return _haversine_meters(lat, lon, next_stop[0], next_stop[1])


def _safe_label_encode(label_encoders: dict, key: str, value: str, default: int = 0) -> int:
    """Encodes `value` with label_encoders[key]; falls back to `default` if the
    encoder is missing or the value is unseen (e.g. an unknown category)."""
    encoder = label_encoders.get(key)
    if encoder is None:
        return default
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return default


# ── prediction logic ───────────────────────────────────────────────────

def _predict_occupancy(payload: BusPayload, models: dict) -> tuple[str, float]:
    """
    Runs the occupancy XGBoost model (trained with SUNT OD).
    Returns (occupancy_class, occupancy_pct).

    If the model is not loaded (xgb_model is None), returns a dummy
    class derived from the raw occupancy value sent by the simulator.

    direction_id and stop_id have no live signal in BusPayload and are held
    at fixed defaults; pt_sequence is approximated with payload.stop_index.
    See the module docstring for details.
    """
    xgb_model          = models.get("xgb_model")
    xgb_encoder        = models.get("xgb_encoder")
    xgb_features       = models.get("xgb_features", [])
    label_encoders     = models.get("label_encoders", {})
    route_loading_mean = models.get("route_loading_mean", {})
    bus_history        = models.setdefault("bus_history", {})

    # ── dummy mode (no model loaded yet) ──────────────────────────────────
    if xgb_model is None:
        occ = payload.occupancy  # 0-100 raw value from the simulator
        if occ < 25:
            occ_class = "low"
        elif occ < 50:
            occ_class = "medium"
        elif occ < 82:
            occ_class = "high"
        else:
            occ_class = "very_high"
        logger.debug("Occupancy model not loaded — dummy mode (occ=%d → %s)", occ, occ_class)
        return occ_class, PCT_MAP[occ_class]

    now = datetime.fromisoformat(payload.timestamp)

    row = {f: 0 for f in xgb_features}

    row["hour"]         = now.hour
    row["day_of_week"]  = payload.day_of_week
    row["is_rush_hour"] = payload.is_rush_hour
    row["is_weekend"]   = int(payload.day_of_week >= 5)

    # Training's route_progression is 0-100 (pt_sequence / max(pt_sequence) * 100
    # per trip); the simulator's route_progress (0-1) is the closest live proxy.
    route_progression = max(0.0, min(100.0, payload.route_progress * 100))
    row["route_progression"] = route_progression

    if route_progression <= 25:
        stage = "start"
    elif route_progression >= 75:
        stage = "end"
    else:
        stage = "middle"
    row["trip_stage"] = _safe_label_encode(label_encoders, "trip_stage", stage)

    if now.hour < 6:
        tod = "night"
    elif now.hour <= 11:
        tod = "morning"
    elif now.hour >= 18:
        tod = "evening"
    else:
        tod = "afternoon"
    row["time_of_day"] = _safe_label_encode(label_encoders, "time_of_day", tod)

    # route_short_name needs to go through the training LabelEncoder
    if "route_short_name" in row and "route_short_name" in label_encoders:
        row["route_short_name"] = _safe_label_encode(label_encoders, "route_short_name", payload.route)

    # loading_mean_route: static per-route average (raw passenger count) from training.
    if route_loading_mean:
        fallback_mean = sum(route_loading_mean.values()) / len(route_loading_mean)
    else:
        fallback_mean = 0.0
    row["loading_mean_route"] = route_loading_mean.get(row["route_short_name"], fallback_mean)

    # loading_lag_1/loading_lag_2: raw occupancy at the previous 1/2 stops for
    # this bus+route, tracked in-memory across requests. Cold start falls back
    # to loading_mean_route rather than 0 (these two features carry 93.6% of
    # the model's importance, so a 0 default would badly bias early predictions).
    hist_key = f"{payload.bus_id}:{payload.route}"
    hist = bus_history.setdefault(hist_key, deque(maxlen=2))
    cold_start_default = row["loading_mean_route"]
    row["loading_lag_1"] = hist[-1] if len(hist) >= 1 else cold_start_default
    row["loading_lag_2"] = hist[-2] if len(hist) >= 2 else cold_start_default
    hist.append(payload.occupancy)

    # No live signal available for these — fixed, documented defaults.
    row["direction_id"] = _safe_label_encode(label_encoders, "direction_id", "I", default=0)
    row["pt_sequence"]  = payload.stop_index
    row["stop_id"]      = 0

    X = pd.DataFrame([row])[xgb_features]
    pred_encoded = xgb_model.predict(X)[0]

    if hasattr(xgb_encoder, "inverse_transform"):
        occ_class = str(xgb_encoder.inverse_transform([pred_encoded])[0])
    else:
        occ_class = OCC_LABELS.get(int(pred_encoded), "medium")

    return occ_class, PCT_MAP.get(occ_class, 50.0)


def _predict_eta(payload: BusPayload, models: dict) -> float:
    """
    Runs the ETA model (XGBoost/RF trained with MTA).
    Returns the estimated minutes until the next stop.

    If the model is not loaded (eta_model is None), it returns a dummy
    value based on route_progress — same behavior as the original main.py.

    Features built here must exactly match those from training.
    The final order is enforced with [eta_features] when assembling
    the DataFrame, just like in the original main.py.
    """
    eta_model    = models.get("eta_model")
    eta_features = models.get("eta_features", [])

    if eta_model is None:
        return round((1 - payload.route_progress) * 5, 2)

    now = datetime.fromisoformat(payload.timestamp)

    distance     = _dist_to_next_stop(payload.lat, payload.lon, payload.stop_index)
    terminal     = BUS_STOPS[-1]
    dist_to_dest = _haversine_meters(payload.lat, payload.lon, terminal[0], terminal[1])

    speed        = max(0.0, min(60.0, payload.speed_kmh))
    # speed_roll3: average with the previous speed if it were available.
    # The simulator does not send previous speed, so we use speed directly
    # (same criterion as the original main.py with prev_speed=None).
    speed_roll3  = speed

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


# ── endpoints ──────────────────────────────────────────────────────────────

@router.post(
    "/predict/eta",
    response_model=PredictionResponse,
    summary="Predicts ETA and occupancy level",
)
async def predict(payload: BusPayload, request: Request):
    """
    Receives current bus data and returns:
    - occupancy_class: occupancy level predicted by XGBoost (SUNT OD)
    - eta_minutes: minutes until the next stop predicted by XGBoost/RF (MTA)

    Used by the simulator and (in the future) by the actual Raspberry Pi.
    """
    models = request.app.state.models
    models["last_payload"] = payload  # latest known bus telemetry, used by /predict/trip

    occ_class, occ_pct = _predict_occupancy(payload, models)
    eta                = _predict_eta(payload, models)

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


@router.post(
    "/predict/eta/broadcast",
    response_model=PredictionResponse,
    summary="Predicts and broadcasts via WebSocket",
)
async def predict_and_broadcast(payload: BusPayload, request: Request):
    """
    Same as POST /predict/eta, but also sends the result
    to all connected WebSocket clients on /ws.

    This is the endpoint used by the simulator (BACKEND_URL = .../predict/eta/broadcast).
    The frontend receives updates via WebSocket without polling.
    """
    result = await predict(payload, request)
    logger.debug(
        "Broadcast → %d WS client(s) | occ=%s eta=%.1fmin",
        len(manager.active), result.occupancy_class, result.eta_minutes,
    )
    await manager.broadcast(result.dict())
    return result


@router.post(
    "/predict/trip",
    response_model=TripResponse,
    summary="Predicts arrival time for an origin-destination trip",
)
async def predict_trip(payload: TripRequest, request: Request):
    """
    Given an origin and destination stop (indices into BUS_STOPS), estimates:
    - eta_to_origin_min: reuses _predict_eta() against the last known bus
      telemetry (from /predict/eta or /predict/eta/broadcast), pretending
      the bus's next stop is origin_stop_id.
    - trip_duration_min: sum of the average segment times (SEGMENTS table)
      between origin and destination, scaled by
      (historical avg speed of those segments / bus's current speed_kmh).
    - total_arrival_time: now + eta_to_origin_min + trip_duration_min.
    """
    n_stops = len(BUS_STOPS)
    if not (0 <= payload.origin_stop_id < n_stops) or not (0 <= payload.destination_stop_id < n_stops):
        raise HTTPException(status_code=400, detail=f"stop ids must be between 0 and {n_stops - 1}")
    if payload.destination_stop_id < payload.origin_stop_id:
        raise HTTPException(status_code=400, detail="destination_stop_id must be at or after origin_stop_id")

    models = request.app.state.models
    last_payload: BusPayload | None = models.get("last_payload")

    # ── eta_to_origin_min: reuse the existing ETA model ─────────────────────
    if last_payload is not None:
        origin_leg_payload = last_payload.copy(
            update={"stop_index": (payload.origin_stop_id - 1) % n_stops}
        )
        eta_to_origin_min = _predict_eta(origin_leg_payload, models)
        current_speed_kmh = last_payload.speed_kmh
    else:
        # No telemetry received yet — fall back to the same dummy ETA used
        # by _predict_eta() when the model itself isn't loaded.
        eta_to_origin_min = 2.0
        current_speed_kmh = 0.0

    # ── trip_duration_min: sum of segment table between origin and destination ──
    leg_segments = SEGMENTS[payload.origin_stop_id:payload.destination_stop_id]
    base_duration_min = sum(seg["avg_time_min"] for seg in leg_segments)
    total_distance_m  = sum(seg["distance_m"] for seg in leg_segments)

    if base_duration_min > 0 and current_speed_kmh > 0:
        historical_avg_speed_kmh = (total_distance_m / 1000) / (base_duration_min / 60)
        speed_factor = historical_avg_speed_kmh / current_speed_kmh
        speed_factor = max(0.3, min(3.0, speed_factor))  # avoid unrealistic swings
    else:
        speed_factor = 1.0

    trip_duration_min = round(base_duration_min * speed_factor, 2)

    total_arrival = datetime.now() + timedelta(minutes=eta_to_origin_min + trip_duration_min)

    return TripResponse(
        eta_to_origin_min=round(eta_to_origin_min, 2),
        trip_duration_min=trip_duration_min,
        total_arrival_time=total_arrival.isoformat(),
    )


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint. The frontend connects here when the page loads
    and listens for JSON messages matching the PredictionResponse format.

    The connection remains open as long as the client is on the page.
    When they disconnect (close the tab, lose network), they are
    automatically cleaned up from the active clients list.
    """
    await manager.connect(ws)
    try:
        while True:
            # We keep the connection open by reading incoming messages.
            # The frontend does not currently send data, but receive_text()
            # is necessary to detect disconnections.
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)