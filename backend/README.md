# Backend

FastAPI service. It loads the two trained models at startup, runs inference on
every telemetry payload it receives, serves the geometry and stops of each known
bus line, answers trip-planning questions, broadcasts predictions over a
WebSocket, and serves the frontend as a static file.

## Running it

From the repository root, not from inside this folder — the package imports are
absolute (`backend.routers...`), so the root has to be on the path:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The dashboard is then at <http://localhost:8000> and the interactive API docs at
<http://localhost:8000/docs>.

Run a single worker. Per-bus occupancy history, the last payload per line, and
the Overpass route cache are all kept in process memory, so a second worker
would see a different half of the state on every request.

## Layout

```
backend/
├── main.py             App creation, CORS, model loading, router registration,
│                       static frontend mount
├── routes_config.py    Registry of known lines — see "Adding a line" below
├── routers/
│   ├── predictions.py  Inference endpoints, the trip planner, and the WebSocket
│   ├── routes.py       Route geometry, stop lists, line list, cache control
│   └── health.py       Status endpoint
├── services/
│   ├── route_services.py  Overpass API client, polyline reconstruction, cache
│   └── ws_manager.py      WebSocket connection manager (singleton)
└── data/
    ├── stop_names.json         Stop coordinates and street names, Línea 38
    ├── stop_names_15_1.json    Same, Línea 15-1
    ├── segment_times.json      Average time and distance per stop-to-stop leg
    └── segment_times_15_1.json Same, Línea 15-1
```

The split is deliberate: `routers/` contains everything FastAPI touches,
`services/` contains logic with no FastAPI dependency, and `main.py` does
nothing but wire them together. New endpoints go in a router, never in
`main.py`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict/eta` | Runs both models on one telemetry payload |
| `POST` | `/predict/eta/broadcast` | Same, and pushes the result to every WebSocket client |
| `POST` | `/predict/occupancy/for-viewer` | Recomputes one bus's occupancy using the caller's local clock |
| `POST` | `/predict/trip` | Wait time, ride duration and arrival time for an origin-destination pair |
| `WS` | `/ws` | Live prediction stream for the dashboard |
| `GET` | `/route?line=38` | Geometry, stops and stop names for a known line |
| `GET` | `/route/lines` | The lines this instance knows about |
| `GET` | `/route/{relation_id}` | Geometry of any line, fetched from OpenStreetMap |
| `GET` | `/route/cache` | Relation IDs currently cached in memory |
| `DELETE` | `/route/cache/{relation_id}` | Drops one cache entry, forcing a refetch |
| `GET` | `/health` | Which models loaded, what is cached, last telemetry per line |
| `GET` | `/` | The dashboard (`frontend/index.html`) |

### Prediction

`POST /predict/eta` takes a `BusPayload` — bus id, line, coordinates,
timestamp, passenger count, speed, stop index, travel direction — and returns a
`PredictionResponse` with the occupancy class, its percentage equivalent, the
ETA in minutes, and the name of the next stop.

`/predict/eta/broadcast` is the same call plus a broadcast, and it is what the
simulator posts to. The frontend never polls; it listens on `/ws`.

Both endpoints also record the payload in two in-memory maps: one keyed by bus
id, used to answer questions about a specific bus, and one keyed by line, used
by the trip planner. Only a line's primary bus writes to the second map, so
that two buses on the same line cannot make the planner's ETA flicker between
their positions.

### Occupancy for the viewer

The occupancy model reads `hour`, `day_of_week` and `is_rush_hour` from the
payload, which come from the clock of whichever machine runs the simulator —
unrelated to whoever is looking at the dashboard.
`POST /predict/occupancy/for-viewer` re-runs the same model for one bus using
the caller's own local time instead, so the occupancy block reflects "now" for
the person reading it.

This call deliberately does not append to the lag history. It can fire far more
often than the simulator's five-second tick — once per viewer, on every clock
change — and perturbing `loading_lag_1` and `loading_lag_2` would corrupt the
real prediction.

### Trip planner

`POST /predict/trip` takes an origin and a destination stop index and returns
three numbers.

`eta_to_origin_min` is the part that needs care. The ETA model was trained on
the distance to the immediately next stop, typically a few hundred meters.
Feeding it the distance to a stop several kilometers away would push a tree
model far outside its training range, which silently under-predicts instead of
scaling. So the wait is computed in two parts: the model handles the leg to the
bus's actual next stop, and the empirical segment table in `data/` covers the
remaining stops up to the origin, scaled by the ratio between the historical
average speed of those segments and the bus's current speed.

`trip_duration_min` is the same segment table summed between origin and
destination, scaled the same way. `total_arrival_time` is the current time plus
both.

The endpoint cross-checks the bus's self-reported `stop_index` against the
nearest stop by GPS position and trusts the GPS whenever the two disagree by
more than one stop, because a client that mistracks the return leg can get
stuck on one stop index indefinitely.

### Route geometry

`GET /route` reads the line's raw OpenStreetMap relation from
`raspberry-pi/route_*.json`, rebuilds the polyline in member order, reverses
individual ways where needed to keep the line continuous, subsamples every
second point, and trims the result to the stretch between the first and last
real stop. All of this happens once at import time, so requests are served from
memory. The raw relation usually continues past the last named stop to a depot
or turnaround — for Línea 38 that is 247 of 548 points — and drawing it would
not match what the bus actually traverses.

`GET /route/{relation_id}` does the same reconstruction for any OpenStreetMap
relation, fetched live from the Overpass API. The first call takes a few
seconds; the result is then cached in memory. Two Overpass mirrors are tried in
order. A relation that exists but cannot be reconstructed returns 404; an
Overpass outage returns 502.

## Adding a line

Everything a line needs is declared in one place. Add its files:

- `raspberry-pi/route_<line>.json` — the raw Overpass response for the
  relation, saved as-is
- `backend/data/stop_names_<line>.json` — each stop's index, coordinates and
  street name
- `backend/data/segment_times_<line>.json` — average minutes, distance and
  speed for each stop-to-stop leg

Then add one entry to `LINES` in [routes_config.py](routes_config.py). No other
backend module needs to change: the route endpoints, the trip planner and the
line picker in the frontend all read from that registry. Add a matching entry
to `LINES` in [../raspberry-pi/simulator.py](../raspberry-pi/simulator.py) if
you want the simulator to drive it.

## Model loading

`_load_models()` in [main.py](main.py) runs once on startup and stores
everything in `app.state.models`; routers read it through
`request.app.state.models`, so there are no globals and no cross-imports with
`main.py`.

If a model file is missing the server still starts. The occupancy endpoint then
falls back to deriving a class from the raw passenger count, and the ETA
endpoint returns a value derived from route progress. This keeps the dashboard
usable for frontend work without Git LFS pulled down, and `/health` reports
which models actually loaded.

One artifact is expected to be missing: `route_loading_mean.pkl` is built from
the training parquet, which is not in the repository. Without it the
`loading_mean_route` feature and the cold-start value of the lag features fall
back to zero, and the startup log says so. Rebuild it with
[../ml/occupancy/scripts/build_route_loading_mean_lookup.py](../ml/occupancy/scripts/build_route_loading_mean_lookup.py)
if you have the dataset.

## Known limitations

- Single-worker only, as described above.
- `allow_origins=["*"]` and no authentication on any endpoint, including the
  telemetry one.
- `@app.on_event` is the older FastAPI startup hook; the lifespan API is the
  current equivalent.
- `direction_id` and `stop_id` are sent to the occupancy model as fixed
  defaults and `pt_sequence` as the stop index, because the live payload has no
  GTFS equivalent for them.
- No automated tests.
