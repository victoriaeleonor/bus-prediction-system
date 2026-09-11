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
import math
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from backend.services.route_services import (
    get_route,
    invalidate_cache,
    list_cached_routes,
)
from backend.routes_config import LINES, DEFAULT_LINE, LineConfig

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
    stop_names: Optional[list[Optional[str]]] = None  # street name per stop, same order as `stops`


class CacheStatusResponse(BaseModel):
    cached_relation_ids: list[int]
    count: int


class LineInfo(BaseModel):
    route_id: str
    label: str


# ── static, locally-known lines (loaded only once at import) ─────────────

def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _trim_to_stops(route: list[tuple[float, float]], bus_stops: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Cuts the polyline down to the stretch between the first and last real
    bus stop — the raw OSM relation typically continues past the last named
    stop to an actual depot/turnaround point (247 of Línea 38's 548 points,
    over 20 minutes of simulated travel, lie beyond its last stop). Keeps
    the drawn route consistent with what raspberry-pi/simulator.py's own
    matching _trim_to_stops() actually makes the bus traverse.
    """
    if not route or not bus_stops:
        return route
    first_stop, last_stop = bus_stops[0], bus_stops[-1]

    def nearest_idx(stop):
        return min(range(len(route)), key=lambda i: _haversine_meters(stop[0], stop[1], route[i][0], route[i][1]))

    i_first, i_last = nearest_idx(first_stop), nearest_idx(last_stop)
    if i_first > i_last:
        i_first, i_last = i_last, i_first
    return route[i_first:i_last + 1]


def _load_static_route(line: LineConfig) -> RouteResponse:
    """
    Loads a line's raspberry-pi/route_*.json and reconstructs the polyline
    in relation-member order, same algorithm as route_services.py uses for
    dynamic OSM lookups. Called once per line at import time.
    """
    with open(line.route_file) as f:
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

    decimated = _trim_to_stops(route[::2], line.bus_stops)
    logger.info("%s loaded from %s: %d points", line.label, line.route_file.name, len(decimated))

    return RouteResponse(
        coordinates=[list(p) for p in decimated],
        stops=[list(p) for p in line.bus_stops],
        relation_id=None,
        from_cache=None,
        stop_names=line.stop_names,
    )


# Loaded once at startup — never blocks a request
_STATIC_ROUTES: dict[str, RouteResponse] = {
    route_id: _load_static_route(line) for route_id, line in LINES.items()
}


# ── endpoints ─────────────────────────────────────────────────────────────

@router.get("", response_model=RouteResponse, summary="Fixed route for a known line")
async def get_fixed_route(line: str = Query(DEFAULT_LINE, description="route_id, see GET /route/lines")):
    """
    Returns the geometry of a locally-known line (see routes_config.LINES),
    loaded from its raspberry-pi/route_*.json. Defaults to Line 38 for
    backward compatibility with clients that don't pass `line`.

    Does not make any external HTTP calls — responds instantly from memory.
    """
    return _STATIC_ROUTES.get(line, _STATIC_ROUTES[DEFAULT_LINE])


@router.get(
    "/lines",
    response_model=list[LineInfo],
    summary="Locally-known bus lines (for a line-picker UI)",
)
async def list_lines():
    """Returns {route_id, label} for every line in routes_config.LINES."""
    return [LineInfo(route_id=rid, label=line.label) for rid, line in LINES.items()]


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