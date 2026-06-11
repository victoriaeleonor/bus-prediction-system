"""
routers/health.py
=================
GET /health — server and loaded models status.

Returns:
{
    "status": "ok",
    "models": {
        "occupancy": true,       ← SUNT OD XGBoost model loaded
        "eta": true | false      ← false if the .pkl file was not found at startup
    },
    "cached_routes": [3983243, 3984378]   ← relation_ids currently stored in memory
}

The frontend does not currently consume this endpoint, but it is useful for
monitoring and for verifying system status before making prediction requests.
"""

import logging

from fastapi import APIRouter, Request

from services.route_services import list_cached_routes

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="Server and models status")
async def health(request: Request):
    models = request.app.state.models

    return {
        "status": "ok",
        "models": {
            "occupancy": models.get("xgb_model") is not None,
            "eta":       models.get("eta_model") is not None,
        },
        "cached_routes": list_cached_routes(),
    }