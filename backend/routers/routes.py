"""
routers/routes.py
=================
Endpoints related to bus route geometry.

Exposed Endpoints
-----------------
GET /route
    Returns the hardcoded route for Line 38 (from route_38.json).
    Maintains full compatibility with the current frontend and simulator:
    no existing files need to be modified.

GET /route/{relation_id}
    Returns the geometry of ANY bus line given its OpenStreetMap
    relation_id. Calls route_service, which downloads the data from Overpass
    and caches the result in memory.

    Example:
        GET /route/3983243   → Route referenced by the frontend teammate
        GET /route/3984378   → Line 38 / Line 55 (current simulator relation)

GET /route/cache
    Lists the relation_ids currently cached in memory.
    Useful for debugging and for the /health endpoint.

DELETE /route/cache/{relation_id}
    Invalidates the cache for a specific relation, forcing a
    re-download from Overpass the next time it is requested.
    Useful if the route changed in OSM without needing to restart the server.

Standard Geometry Response (shared by /route and /route/{relation_id})
----------------------------------------------------------------------
{
    "coordinates": [[lat, lon], ...],   ← complete polyline, used to draw the route on the map
    "stops":       [[lat, lon], ...],   ← route stops (may be [] if OSM does not contain them)
    "relation_id": int | null,          ← null for the fixed Line 38 route
    "from_cache":  bool | null          ← null for the fixed route
}

The current frontend only uses "coordinates" and "stops", so the additional
fields are additive and do not break anything.

Frontend Compatibility
----------------------
The frontend calls `fetch('/route')` and passes the response to
drawRoute({ coordinates, stops }).

The GET /route endpoint continues to return exactly that — nothing changes
in the current API contract.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from services.route_services import (
    get_route,
    invalidate_cache,
    list_cached_routes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/route", tags=["route"])


# ── response schemas ──────────────────────────────────────────────────────

class RouteResponse(BaseModel):
    """
    Shape expected by the frontend: coordinates + stops.
    Extra fields (relation_id, from_cache) are ignored by the current
    frontend but remain available for future versions or debugging.
    """
    coordinates: list[list[float]]
    stops: list[list[float]]
    relation_id: Optional[int] = None
    from_cache: Optional[bool] = None


class CacheStatusResponse(BaseModel):
    cached_relation_ids: list[int]
    count: int


# ── fixed Line 38 route (loaded only once when the module is imported) ───

def _load_line38_from_json() -> RouteResponse:
    """
    Loads route_38.json and reconstructs the polyline in the same order
    used by load_route_from_json() in the original main.py.

    Called only once when the server starts (during module import).
    The result is stored in _LINEA38 for all subsequent requests.
    """
    route_file = Path(__file__).parent.parent.parent / "raspberry-pi" / "route_38.json"

    with open(route_file) as f:
        data = json.load(f)

    node_map = {
        el["id"]: (el["lat"], el["lon"])
        for el in data["elements"]
        if el["type"] == "node"
    }
    way_map = {
        el["id"]: el["nodes"]
        for el in data["elements"]
        if el["type"] == "way"
    }
    relation = next(el for el in data["elements"] if el["type"] == "relation")

    route: list[tuple[float, float]] = []
    prev_last: Optional[tuple[float, float]] = None

    for m in relation["members"]:
        if m["type"] != "way" or m["ref"] not in way_map:
            continue
        coords = [node_map[n] for n in way_map[m["ref"]] if n in node_map]
        if not coords:
            continue
        if prev_last:
            d_fwd = abs(coords[0][0] - prev_last[0]) + abs(coords[0][1] - prev_last[1])
            d_rev = abs(coords[-1][0] - prev_last[0]) + abs(coords[-1][1] - prev_last[1])
            if d_rev < d_fwd:
                coords = coords[::-1]
            if min(d_fwd, d_rev) > 0.015:
                continue
        route.extend(coords)
        prev_last = route[-1]

    # Hardcoded stops — same ones used in main.py and simulator.py.
    # These are the actual Line 38 stops used for ETA calculations.
    # If new lines with their own models are added, each should have
    # its stops defined in a configuration file or database.
    bus_stops = [
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

    logger.info("Line 38 loaded from route_38.json: %d points", len(route[::2]))

    return RouteResponse(
        coordinates=[list(p) for p in route[::2]],
        stops=[list(p) for p in bus_stops],
        relation_id=None,
        from_cache=None,
    )


# Loaded once at startup — never blocks a request
_LINEA38: RouteResponse = _load_line38_from_json()


# ── endpoints ─────────────────────────────────────────────────────────────

@router.get("", response_model=RouteResponse, summary="Fixed Line 38 route")
async def get_fixed_route():
    """
    Returns the geometry of Line 38 loaded from route_38.json.

    This is the endpoint currently used by the frontend and simulator.
    It does not make any external HTTP calls — it responds instantly
    from memory. The response contract remains unchanged.
    """
    return _LINEA38


@router.get(
    "/cache",
    response_model=CacheStatusResponse,
    summary="List of routes cached in memory",
)
async def get_cache_status():
    """
    Returns the OSM relation_ids currently stored in cache.
    Useful for the /health endpoint and development debugging.
    """
    cached = list_cached_routes()
    return CacheStatusResponse(cached_relation_ids=cached, count=len(cached))


@router.delete(
    "/cache/{relation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Invalidate a route cache entry",
)
async def delete_route_cache(relation_id: int):
    """
    Forces a re-download from Overpass the next time this relation
    is requested. Useful if the route has changed in OSM.
    """
    invalidate_cache(relation_id)
    logger.info("Cache manually invalidated for relation_id %d", relation_id)


@router.get(
    "/{relation_id}",
    response_model=RouteResponse,
    summary="Geometry of any bus line by OSM relation_id",
)
async def get_dynamic_route(relation_id: int):
    """
    Downloads and returns the geometry of the bus line identified by
    its OpenStreetMap relation_id.

    - First request: downloads from Overpass API (~2–10 sec) and caches.
    - Subsequent requests: served from in-memory cache (<1 ms).

    The `from_cache` field in the response indicates whether the data
    came from cache.

    Possible errors:
    - 404: relation_id does not exist in OSM or is not a route relation.
    - 502: Overpass API did not respond (all servers failed).
    - 422: relation_id is not a valid integer (automatic FastAPI validation).

    Example using the relation referenced by the frontend teammate:
        GET /route/3983243
    """
    try:
        result = await get_route(relation_id)
    except ValueError as exc:
        # OSM returned data but we could not reconstruct a valid route.
        # Most likely the relation exists but is not a bus route with
        # the expected way/node structure.
        logger.warning("ValueError for relation %d: %s", relation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not build a route for relation_id {relation_id}. "
                   f"Verify that it is a 'route' relation in OSM. "
                   f"Details: {exc}",
        )
    except RuntimeError as exc:
        # Overpass did not respond — network issue or server outage.
        logger.error("RuntimeError for relation %d: %s", relation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not contact the Overpass API to download relation {relation_id}. "
                   f"Please try again in a few minutes. Details: {exc}",
        )

    return RouteResponse(
        coordinates=result["coordinates"],
        stops=result["stops"],
        relation_id=result["relation_id"],
        from_cache=result["from_cache"],
    )