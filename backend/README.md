# Backend — Bus Prediction System

FastAPI backend that serves real-time bus ETA and occupancy predictions, exposes route geometry from OpenStreetMap, and broadcasts updates to connected clients via WebSocket.

## Models

| Model | Algorithm | Training Data |
|---|---|---|
| Occupancy | XGBoost | SUNT OD |
| ETA | Random Forest | MTA |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict/eta` | Returns occupancy class + ETA in minutes |
| `POST` | `/predict/eta/broadcast` | Same as above, broadcasts result via WebSocket |
| `GET` | `/route` | Fixed Línea 38 geometry (coordinates + stops) |
| `GET` | `/route/{relation_id}` | Any bus line geometry from OpenStreetMap |
| `GET` | `/route/cache` | Lists relation IDs currently cached in memory |
| `DELETE` | `/route/cache/{relation_id}` | Invalidates cached route, forces re-fetch |
| `WS` | `/ws` | WebSocket connection for real-time updates |
| `GET` | `/health` | Server and model status |

## Folder Structure

```
backend/
├── main.py                  # FastAPI app, middleware, router registration, model loading
├── routers/                 # All HTTP and WebSocket endpoints
│   ├── predictions.py       # POST /predict/eta, POST /predict/eta/broadcast, WS /ws
│   ├── routes.py            # GET /route, GET /route/{relation_id}
│   └── health.py            # GET /health
├── services/                # Business logic, no FastAPI dependencies
│   ├── route_service.py     # Fetches route geometry from Overpass API, caches result
│   └── ws_manager.py        # WebSocket ConnectionManager singleton
└── models/                  # Create if working with multiple bus lines
    └── schemas.py           # Shared Pydantic models (BusPayload, PredictionResponse)
```

> `occupancy_service.py`, `eta_service.py`, and `utils/geo.py` are planned for when prediction logic is extracted out of `predictions.py` (recommended if adding more lines).

## Running the Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs` once the server is running.

## Route Geometry

`GET /route` serves the fixed Línea 38 geometry from `route_38.json` (no external calls).

`GET /route/{relation_id}` fetches any line from the [Overpass API](https://overpass-api.de) using its OpenStreetMap relation ID. The result is cached in memory — first call takes a few seconds, subsequent calls are instant.

```bash
# Example: fetch the relation referenced by the frontend
GET /route/3983243
```

## Simulator (Raspberry Pi)

The simulator lives in `/raspberry-pi/simulator.py`. Before running it on the Pi, update `BACKEND_URL` to point to the machine running the backend:

```python
BACKEND_URL = "http://<backend-machine-ip>:8000/predict/eta/broadcast"
```

Then on the Pi:

```bash
pip install requests
python simulator.py
```

`route_38.json` must be in the same folder as `simulator.py`. If missing, the simulator will attempt to download the route from Overpass on startup.