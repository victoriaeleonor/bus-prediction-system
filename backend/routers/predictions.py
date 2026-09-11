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

import logging
import math
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

#from services.ws_manager import manager
from backend.services.ws_manager import manager
from backend.routes_config import DEFAULT_LINE, get_line

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])

# Each line can now run more than one bus (see raspberry-pi/simulator.py's
# LINES registry — bus_ids[0] is the line's "primary"). /predict/trip and
# the sidebar's per-line state (last_payload_by_route) only ever look at
# the primary bus, so adding a second bus to a line doesn't make that
# single-slot cache flicker between two buses' positions. Every bus's own
# state is still tracked independently in last_payload_by_bus, keyed by
# bus_id, for endpoints/broadcasts that care about a specific bus.
PRIMARY_BUS_ID = {"38": "bus_001", "15-1": "bus_002"}


# ── schemas ────────────────────────────────────────────────────────────────
# Same fields as the original main.py — the simulator does not need changes.

class BusPayload(BaseModel):
    bus_id: str
    route: str
    route_id: str = DEFAULT_LINE  # key into routes_config.LINES ("38", "15-1", ...)
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
    travel_direction: int = 1  # +1 outbound (stop_index -> stop_index+1), -1 on the return leg


class PredictionResponse(BaseModel):
    bus_id: str
    route: str
    route_id: str
    lat: float
    lon: float
    timestamp: str
    occupancy_raw: int
    occupancy_class: str    # "low" | "medium" | "high" | "very_high"
    occupancy_pct: float    # percentage representation of the level
    eta_minutes: float      # minutes until the next stop
    speed_kmh: float
    next_stop: Optional[str] = None  # street name of the next stop, from stop_names.json
    hour: int
    day_of_week: int
    is_rush_hour: int


class OccupancyForViewerRequest(BaseModel):
    bus_id: str
    hour: int          # the VIEWER's local hour (0-23), not the simulator's
    day_of_week: int   # the VIEWER's local day (0=Monday .. 6=Sunday)
    is_rush_hour: int


class OccupancyForViewerResponse(BaseModel):
    bus_id: str
    occupancy_class: str
    occupancy_pct: float


class TripRequest(BaseModel):
    origin_stop_id: int       # index into the line's bus_stops, 0-based
    destination_stop_id: int  # index into the line's bus_stops, 0-based, must be >= origin_stop_id
    route_id: str = DEFAULT_LINE
    bus_id: Optional[str] = None  # which of the line's buses to use; falls back to the line's primary if omitted


class TripResponse(BaseModel):
    eta_to_origin_min: float   # minutes for the bus to reach origin_stop_id
    trip_duration_min: float   # minutes from origin_stop_id to destination_stop_id
    total_arrival_time: str    # ISO timestamp: now + eta_to_origin + trip_duration
    bus_already_passed: bool = False  # True if the bus already went past origin_stop_id


OCC_LABELS = {0: "low", 1: "medium", 2: "high", 3: "very_high"}
PCT_MAP    = {"low": 15.0, "medium": 40.0, "high": 65.0, "very_high": 88.0}


# ── geometric helpers ────────────────────────────────────────────────────

def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _dist_to_next_stop(lat: float, lon: float, stop_index: int, direction: int, bus_stops: list) -> float:
    """Distance in meters to the next stop (stop_index + direction).

    direction is +1 while the bus travels outbound and -1 on the return leg
    (see BusPayload.travel_direction) — without it, "next stop" would always
    be assumed to be stop_index + 1, which is wrong for half of a round trip.
    """
    next_idx = (stop_index + direction) % len(bus_stops)
    next_stop = bus_stops[next_idx]
    return _haversine_meters(lat, lon, next_stop[0], next_stop[1])


def _next_stop_name(stop_index: int, direction: int, stop_names: list) -> Optional[str]:
    """Street name of the next stop (stop_index + direction), for display."""
    next_idx = (stop_index + direction) % len(stop_names)
    return stop_names[next_idx]


def _nearest_stop_index(lat: float, lon: float, bus_stops: list) -> int:
    """Index of the bus_stops entry geographically closest to (lat, lon).

    Used as a GPS-based cross-check against the bus's self-reported
    stop_index, which can drift from reality (e.g. a client that fails to
    track the return leg correctly).
    """
    return min(range(len(bus_stops)), key=lambda i: _haversine_meters(lat, lon, *bus_stops[i]))


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

def _predict_occupancy(payload: BusPayload, models: dict, record_history: bool = True) -> tuple[str, float]:
    """
    Runs the occupancy XGBoost model (trained with SUNT OD).
    Returns (occupancy_class, occupancy_pct).

    If the model is not loaded (xgb_model is None), returns a dummy
    class derived from the raw occupancy value sent by the simulator.

    direction_id and stop_id have no live signal in BusPayload and are held
    at fixed defaults; pt_sequence is approximated with payload.stop_index.
    See the module docstring for details.

    record_history=False skips appending to loading_lag's history deque —
    used by /predict/occupancy/for-viewer, which re-runs this prediction
    with a different `hour`/`day_of_week`/`is_rush_hour` purely for display
    and must not advance the real simulator-driven lag tracking every time
    a viewer's browser clock ticks over.
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
    hist_key = f"{payload.bus_id}:{payload.route_id}"
    hist = bus_history.setdefault(hist_key, deque(maxlen=2))
    cold_start_default = row["loading_mean_route"]
    row["loading_lag_1"] = hist[-1] if len(hist) >= 1 else cold_start_default
    row["loading_lag_2"] = hist[-2] if len(hist) >= 2 else cold_start_default
    if record_history:
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
    bus_stops = get_line(payload.route_id).bus_stops

    distance     = _dist_to_next_stop(payload.lat, payload.lon, payload.stop_index, payload.travel_direction, bus_stops)
    terminal     = bus_stops[-1] if payload.travel_direction >= 0 else bus_stops[0]
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
    # Every bus's own latest telemetry, keyed by bus_id — powers the
    # occupancy-by-selected-bus feature without touching last_payload_by_route.
    models.setdefault("last_payload_by_bus", {})[payload.bus_id] = payload

    # Latest known telemetry per line, used by /predict/trip — keyed by
    # route_id so two lines running concurrently don't clobber each other.
    # Only the line's primary bus writes here: with two buses per line now,
    # letting either one overwrite this single slot would make the trip
    # planner's ETA flicker between two different buses' positions.
    if payload.bus_id == PRIMARY_BUS_ID.get(payload.route_id, payload.bus_id):
        models.setdefault("last_payload_by_route", {})[payload.route_id] = payload

    line = get_line(payload.route_id)
    occ_class, occ_pct = _predict_occupancy(payload, models)
    eta                = _predict_eta(payload, models)
    next_stop          = _next_stop_name(payload.stop_index, payload.travel_direction, line.stop_names)

    return PredictionResponse(
        bus_id          = payload.bus_id,
        route           = payload.route,
        route_id        = payload.route_id,
        lat             = payload.lat,
        lon             = payload.lon,
        timestamp       = payload.timestamp,
        occupancy_raw   = payload.occupancy,
        occupancy_class = occ_class,
        occupancy_pct   = occ_pct,
        eta_minutes     = eta,
        speed_kmh       = payload.speed_kmh,
        next_stop       = next_stop,
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
    "/predict/occupancy/for-viewer",
    response_model=OccupancyForViewerResponse,
    summary="Recomputes a bus's occupancy using the viewer's local time",
)
async def predict_occupancy_for_viewer(payload: OccupancyForViewerRequest, request: Request):
    """
    The occupancy model's hour/day_of_week/is_rush_hour features normally
    come from the simulator's own machine clock — whoever happens to be
    running raspberry-pi/simulator.py, which has nothing to do with
    whoever is looking at the dashboard. This re-runs the same model for
    one bus using the caller's own local time instead, so the OCUPACIÓN
    block reflects "now" for whoever opened the page, not for wherever
    the simulator happens to be running.

    record_history=False: this can be called far more often than the
    simulator's real 5s tick (e.g. once per viewer, on every clock change),
    and must not perturb the real loading_lag_1/2 tracking used by the
    actual simulator-driven prediction in /predict/eta.
    """
    models = request.app.state.models
    base = models.get("last_payload_by_bus", {}).get(payload.bus_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"No telemetry yet for bus_id '{payload.bus_id}'")

    viewer_payload = base.copy(update={
        "hour": payload.hour,
        "day_of_week": payload.day_of_week,
        "is_rush_hour": payload.is_rush_hour,
    })
    occ_class, occ_pct = _predict_occupancy(viewer_payload, models, record_history=False)

    return OccupancyForViewerResponse(bus_id=payload.bus_id, occupancy_class=occ_class, occupancy_pct=occ_pct)


@router.post(
    "/predict/trip",
    response_model=TripResponse,
    summary="Predicts arrival time for an origin-destination trip",
)
async def predict_trip(payload: TripRequest, request: Request):
    """
    Given an origin and destination stop (indices into the line's bus_stops),
    estimates:
    - eta_to_origin_min: _predict_eta() only covers the leg from the bus's
      current position to its real next stop (the model was trained on
      "distance to the immediately next stop", generally a few hundred
      meters — feeding it a multi-km distance to a faraway origin_stop_id
      would extrapolate a tree model far outside its training range, which
      silently under-predicts instead of scaling proportionally). Any
      remaining stops between the bus's real next stop and origin_stop_id
      are covered with the same segment table used for trip_duration_min.
    - trip_duration_min: sum of the average segment times (the line's
      segment table) between origin and destination, scaled by
      (historical avg speed of those segments / bus's current speed_kmh).
    - total_arrival_time: now + eta_to_origin_min + trip_duration_min.
    """
    line = get_line(payload.route_id)
    bus_stops = line.bus_stops
    segments = line.segments

    n_stops = len(bus_stops)
    if not (0 <= payload.origin_stop_id < n_stops) or not (0 <= payload.destination_stop_id < n_stops):
        raise HTTPException(status_code=400, detail=f"stop ids must be between 0 and {n_stops - 1}")
    if payload.destination_stop_id < payload.origin_stop_id:
        raise HTTPException(status_code=400, detail="destination_stop_id must be at or after origin_stop_id")

    models = request.app.state.models
    if payload.bus_id:
        # A specific bus was picked (e.g. clicked on the map) — use its own
        # telemetry rather than always the line's primary bus.
        last_payload: Optional[BusPayload] = models.get("last_payload_by_bus", {}).get(payload.bus_id)
    else:
        last_payload = models.get("last_payload_by_route", {}).get(payload.route_id)

    def _segments_duration_m(from_stop: int, to_stop: int) -> tuple[float, float]:
        """Sum of (avg_time_min, distance_m) for segments[from_stop:to_stop]."""
        legs = segments[from_stop:to_stop]
        return sum(s["avg_time_min"] for s in legs), sum(s["distance_m"] for s in legs)

    def _speed_adjusted_minutes(base_minutes: float, distance_m: float, current_speed_kmh: float) -> float:
        if base_minutes <= 0 or current_speed_kmh <= 0:
            return base_minutes
        historical_avg_speed_kmh = (distance_m / 1000) / (base_minutes / 60)
        factor = max(0.3, min(3.0, historical_avg_speed_kmh / current_speed_kmh))  # avoid unrealistic swings
        return base_minutes * factor

    bus_already_passed = False

    if last_payload is not None:
        current_speed_kmh = last_payload.speed_kmh
        direction = last_payload.travel_direction if last_payload.travel_direction in (1, -1) else 1

        # The client's self-reported stop_index can drift from reality (a
        # client that doesn't track the return leg correctly will get stuck
        # at one stop forever — see raspberry-pi/simulator.py history). Cross
        # -check it against the nearest stop by actual GPS position and
        # trust the GPS whenever they disagree by more than one stop.
        reported_stop = last_payload.stop_index % n_stops
        gps_stop = _nearest_stop_index(last_payload.lat, last_payload.lon, bus_stops)
        drift = min((reported_stop - gps_stop) % n_stops, (gps_stop - reported_stop) % n_stops)
        current_stop = gps_stop if drift > 1 else reported_stop

        bus_next_stop = (current_stop + direction) % n_stops  # real next stop, matches _predict_eta's training assumption

        # Signed count of stops between the bus's real next stop and the
        # chosen origin, positive when origin is still ahead — regardless of
        # travel direction (+1 outbound / -1 on the return leg).
        steps_ahead = (payload.origin_stop_id - bus_next_stop) * direction

        if steps_ahead < 0:
            # The bus already passed this stop — no ETA to compute.
            eta_to_origin_min = 0.0
            bus_already_passed = True
        elif steps_ahead == 0:
            # origin is exactly the bus's real next stop — squarely inside
            # the model's trained scenario, use it directly.
            eta_to_origin_min = _predict_eta(last_payload, models)
        else:
            # ETA to the bus's real next stop (valid model usage) + the
            # empirical segment table for the remaining stops up to origin
            # (kept outside the ML model's input range, no extrapolation).
            eta_to_next = _predict_eta(last_payload, models)
            lo, hi = sorted((bus_next_stop, payload.origin_stop_id))
            base_min, dist_m = _segments_duration_m(lo, hi)
            eta_to_origin_min = eta_to_next + _speed_adjusted_minutes(base_min, dist_m, current_speed_kmh)
    else:
        # No telemetry received yet — fall back to the same dummy ETA used
        # by _predict_eta() when the model itself isn't loaded.
        eta_to_origin_min = 2.0
        current_speed_kmh = 0.0

    # ── trip_duration_min: sum of segment table between origin and destination ──
    base_duration_min, total_distance_m = _segments_duration_m(payload.origin_stop_id, payload.destination_stop_id)
    trip_duration_min = round(_speed_adjusted_minutes(base_duration_min, total_distance_m, current_speed_kmh), 2)

    total_arrival = datetime.now() + timedelta(minutes=eta_to_origin_min + trip_duration_min)

    return TripResponse(
        eta_to_origin_min=round(eta_to_origin_min, 2),
        trip_duration_min=trip_duration_min,
        total_arrival_time=total_arrival.isoformat(),
        bus_already_passed=bus_already_passed,
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