"""
main.py
=======
FastAPI server entry point.

Responsibilities of this file (and only these):
  1. Create the FastAPI app and configure middleware (CORS).
  2. Load ML models only once at startup and store them in app.state,
     so routers can access them without re-importing or using global variables.
  3. Register routers (HTTP and WebSocket routes).
  4. Serve the static frontend.

Everything else (prediction logic, route reconstruction, WebSocket manager)
lives in routers/ and services/. If you need to add a new endpoint,
do not modify this file — create or edit the corresponding router instead.

Starting the server
-------------------
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import pickle
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── routers ───────────────────────────────────────────────────────────────
# Each router is responsible for its own endpoints.
# main.py simply mounts them — it doesot need to know what is inside


from backend.routers.routes import router as routes_router
from backend.routers.predictions import router as predictions_router
from backend.routers.health import router as health_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── ML model paths ────────────────────────────────────────────────────────
# BASE points to the repository root (one level above /backend).
BASE    = Path(__file__).parent.parent
OCC_DIR = BASE / "ml" / "occupancy" / "models"
ETA_DIR = BASE / "ml" / "eta" / "model"
SUFFIX  = "with_lags"   # suffix used by the occupancy model .pkl files


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_models() -> dict:
    """
    Loads all ML models and returns them in a dictionary.

    This function is called only once during the FastAPI startup event.
    The dictionary is stored in app.state.models so routers can access it
    via request.app.state.models — without cross-imports or global variables.

    If the ETA model does not exist (file not found), None is stored
    and prediction endpoints fall back to dummy mode, just like before.
    """
    models = {}

    # Occupancy (XGBoost, trained on the SUNT OD dataset) 
    logger.info("Loading occupancy model...")
    models["xgb_model"] = _load_pkl(OCC_DIR / f"xgboost_occupancy_{SUFFIX}.pkl")
    models["xgb_encoder"] = _load_pkl(OCC_DIR / f"xgboost_label_encoder_{SUFFIX}.pkl")
    models["xgb_features"] = _load_pkl(OCC_DIR / f"xgboost_feature_names_{SUFFIX}.pkl")
    models["label_encoders"] = _load_pkl(
        BASE / "ml" / "occupancy" / "data" / f"sunt_2024_03_{SUFFIX}_encoders.pkl"
    )
    logger.info("✓ Occupancy model loaded (%d features)", len(models["xgb_features"]))

    # ETA (XGBoost/RF, trained on the MTA dataset) 
    logger.info("Loading ETA model...")
    try:
        models["eta_model"] = joblib.load(ETA_DIR / "xgb_model_eta_final.pkl")
        eta_meta = _load_pkl(ETA_DIR / "xgb_features_eta_final.pkl")
        models["eta_features"] = eta_meta["features"]
        logger.info(
            "✓ ETA model loaded (%d features): %s",
            len(models["eta_features"]),
            models["eta_features"],
        )
    except FileNotFoundError:
        models["eta_model"]    = None
        models["eta_features"] = []
        logger.warning("⚠ ETA model not found — predictions will use dummy mode.")

    return models


# ── app ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bus Prediction API",
    description=(
        "Backend for the bus ETA and occupancy prediction system. "
        "Occupancy: XGBoost trained on SUNT OD. "
        "ETA: XGBoost/RF trained on MTA."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── lifecycle events ──────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Runs only once when uvicorn starts the server.
    Loads the models and stores them in app.state so all
    routers can use them without reloading or relying on globals.
    """
    logger.info("=== Starting Bus Prediction API ===")
    try:
        app.state.models = _load_models()
    except FileNotFoundError as e:
        logger.warning("Models not found, starting without ML: %s", e)
        app.state.models = {
            "xgb_model": None,
            "xgb_encoder": None,
            "xgb_features": [],
            "label_encoders": {},
            "eta_model": None,
            "eta_features": [],
        }
    logger.info("=== Server ready ===")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=== Server stopped ===")


# ── routers ───────────────────────────────────────────────────────────────
# The order here defines priority in case of overlapping prefixes,
# although each router has a different prefix so there is no conflict.

app.include_router(routes_router)       # GET  /route, /route/{relation_id}, etc.
app.include_router(predictions_router)  # POST /predict/eta, POST /predict/eta/broadcast, WS /ws
app.include_router(health_router)       # GET  /health


# ── static frontend ───────────────────────────────────────────────────────
# Serves index.html and frontend assets from /frontend.
# This remains unchanged from the original main.py.

_frontend_path = Path(__file__).parent.parent / "frontend"

if _frontend_path.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(_frontend_path)),
        name="static",
    )

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(_frontend_path / "index.html"))
else:
    logger.warning(
        "Frontend directory not found at %s — "
        "the server will run but will not serve the UI.",
        _frontend_path,
    )