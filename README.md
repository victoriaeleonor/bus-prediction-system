# Bus Prediction System

A real-time bus tracking dashboard for Asunción, Paraguay. It shows where a bus
is right now, how full it is, when it will reach a given stop, and how long a
trip between two stops will take.

The system is made of four pieces that talk to each other:

```
raspberry-pi/simulator.py          backend/ (FastAPI)              frontend/index.html
  GPS position, speed,      POST    ML inference                WS   Leaflet map,
  passenger count, clock   ───────► occupancy + ETA models   ───────► sidebar, trip
  every 5 seconds                   route geometry, stops            planner
                                           │
                                           │ loads at startup
                                           ▼
                                    ml/ (trained .pkl artifacts)
```

- **`raspberry-pi/`** produces the telemetry. Today a Python simulator drives
  buses along the real OpenStreetMap geometry of each line; on real hardware
  the same payload would come from a GPS unit and a passenger counter.
- **`backend/`** is a FastAPI service. It loads the two trained models once at
  startup, runs inference on every incoming payload, serves route geometry, and
  pushes each prediction to every connected browser over a WebSocket.
- **`frontend/`** is a single self-contained HTML file. No build step, no
  framework, no bundler — the backend serves it directly.
- **`ml/`** holds the notebooks that trained the models and the exported
  artifacts the backend loads.

Each folder has its own README with the details.

## What the dashboard does

- Live map of two real lines, Línea 38 and Línea 15-1, with several buses
  running on each and colored by travel direction (outbound or return leg).
- Occupancy prediction per bus (low, medium, high, very high), recomputed
  against the viewer's own local clock rather than the simulator's.
- ETA to the bus's next stop, updated every five seconds.
- "When does my bus arrive?" — pick a stop from the list, use browser
  geolocation, or tap a point on the map, and get the arrival time for the
  nearest stop.
- Trip planner — pick an origin and a destination stop and get the wait, the
  ride duration, and the estimated arrival time.
- Spanish and English, switchable at any time without reloading.
- Works on a phone: the sidebar collapses into an overlay and the layout tracks
  the real visible viewport height on iOS Safari.

## Models

| | Occupancy | ETA |
|---|---|---|
| Algorithm | XGBoost (multiclass) | XGBoost (regression) |
| Dataset | SUNT OD, Salvador, Brazil | MTA bus telemetry, New York |
| Training size | 14.6M stop records (March 2024) | 2.1M records (June 2017) |
| Result | 94.0% accuracy, 0.916 macro F1 | MAE 18.1 s, R² 0.919 |
| Output | low / medium / high / very high | minutes to next stop |

Full feature lists, training procedure and evaluation are in
[ml/README.md](ml/README.md).

## Running it locally

You need Python 3.9 or newer and [Git LFS](https://git-lfs.com), since the model
files are stored through LFS.

```bash
git lfs install
git clone <repository-url>
cd bus-prediction-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the backend from the repository root:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then, in a second terminal, start the simulator so there is something to show:

```bash
pip install requests
python raspberry-pi/simulator.py --line 38
```

Open <http://localhost:8000>. Interactive API docs are at
<http://localhost:8000/docs>.

To run both lines at once, start a second simulator with `--line 15-1`.

## Deployment

The service is deployed on Render from [render.yaml](render.yaml), which builds
from the `deploy/render` branch and serves both the API and the frontend from a
single web service. The simulator is not deployed — it runs wherever you choose
and points at the deployed instance:

```bash
python raspberry-pi/simulator.py --line 38 \
  --backend-url https://<your-app>.onrender.com/predict/eta/broadcast
```

On Render's free plan the instance sleeps after inactivity, so the first request
after a pause takes several seconds while the models reload.

## Repository layout

```
.
├── backend/              FastAPI service — see backend/README.md
│   ├── main.py           App creation, model loading, router registration
│   ├── routes_config.py  Registry of known bus lines and their data files
│   ├── routers/          HTTP and WebSocket endpoints
│   ├── services/         Overpass route fetching, WebSocket manager
│   └── data/             Stop names and segment travel times per line
├── frontend/             Single-file dashboard — see frontend/README.md
├── ml/                   Notebooks and trained models — see ml/README.md
│   ├── occupancy/        SUNT OD pipeline, XGBoost and Random Forest
│   └── eta/              MTA pipeline, XGBoost, Random Forest, CNN experiments
├── raspberry-pi/         Telemetry source — see raspberry-pi/README.md
│   ├── simulator.py      Bus GPS simulator
│   └── route_*.json      Raw OpenStreetMap relation data per line
├── render.yaml           Render deployment definition
└── requirements.txt      Runtime dependencies for the backend
```

## Project status

The system is feature-complete for a beta release. The full path from telemetry
to prediction to browser works end to end, both models are trained and
evaluated, the deployment is live, and the interface covers what a rider
actually needs. What follows are the known limitations, stated plainly so that
beta testers and reviewers know what they are looking at.

**The data is simulated, not measured.** No GPS hardware or passenger counter is
connected yet. The simulator interpolates positions along the real route
geometry and generates passenger counts from time-of-day ranges, so the
occupancy numbers on screen are plausible rather than observed.

**The ETA model was trained on New York, not Asunción.** Its features are
generic — distance, speed, hour of day — which is why it transfers at all, but
absolute times will not match local traffic behavior until it is retrained on
Paraguayan data.

**Two occupancy features have no live signal.** `direction_id` and `stop_id`
are held at fixed defaults and `pt_sequence` is approximated from the stop
index, because the live payload has no GTFS equivalent. Their combined
importance during training was under 2%, so the effect is small but real.

**`route_loading_mean.pkl` is not committed.** It is built from the training
parquet, which lives outside the repository, so a fresh deployment logs a
warning and falls back to zero for `loading_mean_route` and for the cold-start
value of the lag features. See
[ml/occupancy/scripts/build_route_loading_mean_lookup.py](ml/occupancy/scripts/build_route_loading_mean_lookup.py).

**Server state is in memory and single-worker.** Per-bus occupancy history, the
last payload per line, and the Overpass route cache all live in the process.
Running more than one uvicorn worker would split that state, and a restart
clears it.

**There are no automated tests.** Verification so far has been manual, against
the running dashboard.

**CORS is fully open** (`allow_origins=["*"]`) and no endpoint requires
authentication. That is acceptable for a public read-only demo, but
`/predict/eta/broadcast` accepts telemetry from anyone who finds it.

## Roadmap

The natural next steps, roughly in order of value:

1. Replace the simulator with a real GPS unit and passenger counter on a bus.
2. Retrain the ETA model on locally collected data.
3. Add a test suite covering the prediction endpoints and the trip planner math.
4. Move in-memory state to Redis so the service can run more than one worker.
5. Restrict CORS and authenticate the telemetry endpoint before any deployment
   beyond a demo.
6. Add the remaining lines, which is a matter of adding data files and one entry
   to `routes_config.LINES`.
