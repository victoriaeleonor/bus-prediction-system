"""
routers/routes.py
=================
Endpoints related to bus route geometry.

Exposed endpoints
-----------------
GET /route
    Returns the hardcoded route of Line 38 (from route_38.json).
    Maintains full compatibility with the current frontend and simulator:
    no existing files need to be modified.

GET /route/{relation_id}
    Returns the geometry of ANY bus line given its OpenStreetMap
    relation_id. Calls route_service, which downloads data from Overpass
    and caches the result in memory.

    Example:
        GET /route/3983243   → Line referenced by the frontend teammate
        GET /route/3984378   → Line 38 / Line 55 (current simulator relation)

GET /route/cache
    Lists the relation_ids currently cached in memory.
    Useful for debugging and for the /health endpoint.

DELETE /route/cache/{relation_id}
    Invalidates the cache for a specific relation, forcing a
    re-download from Overpass the next time it is requested.
    Useful if the route changed in OSM without needing to restart the server.

Standard geometry response (shared by /route and /route/{relation_id})
----------------------------------------------------------------------
{
    "coordinates": [[lat, lon], ...],   ← complete polyline, used to draw the route on the map
    "stops":       [[lat, lon], ...],   ← bus stops (may be [] if OSM does not contain them)
    "relation_id": int | null,          ← null for the fixed Line 38 route
    "from_cache":  bool | null          ← null for the fixed route
}

The current frontend only uses "coordinates" and "stops", so the additional
fields are additive and do not break anything.

Frontend compatibility
----------------------
The frontend calls `fetch('/route')` and passes the response to
drawRoute({ coordinates, stops }).
The GET /route endpoint continues returning exactly that — nothing changes
in the current API contract.
"""

# ── response schemas ──────────────────────────────────────────────────────

class RouteResponse(BaseModel):
    """
    Format expected by the frontend: coordinates + stops.
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

def _load_linea38_from_json() -> RouteResponse:
    """
    Loads route_38.json and rebuilds the polyline in the same order
    as load_route_from_json() did in the original main.py.

    Called only once when the server starts (when this module is imported).
    The result is stored in _LINEA38 for all subsequent requests.
    """

    ...

    # Hardcoded bus stops — same ones used in main.py and simulator.py.
    # These are the real Line 38 stops used for ETA calculation.
    # If new lines are added with their own models, each one should
    # have its stops in a configuration file or database.

    ...

    logger.info("Line 38 loaded from route_38.json: %d points", len(route[::2]))

    ...


# Loaded once at startup — never blocks a request
_LINEA38: RouteResponse = _load_linea38_from_json()


# ── endpoints ─────────────────────────────────────────────────────────────

@router.get("", response_model=RouteResponse, summary="Fixed Line 38 Route")
async def get_fixed_route():
    """
    Returns the geometry of Line 38 loaded from route_38.json.

    This is the endpoint currently used by the frontend and simulator.
    It does not make any external HTTP calls — it responds instantly
    from memory. Its response contract does not change.
    """
    return _LINEA38


@router.get(
    "/cache",
    response_model=CacheStatusResponse,
    summary="List of routes cached in memory",
)
async def get_cache_status():
    """
    Returns the OSM relation_ids that are currently cached.
    Useful for the /health endpoint and for development debugging.
    """
    cached = list_cached_routes()
    return CacheStatusResponse(cached_relation_ids=cached, count=len(cached))


@router.delete(
    "/cache/{relation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Invalidate a relation cache",
)
async def delete_route_cache(relation_id: int):
    """
    Forces a re-download from Overpass the next time this relation
    is requested. Useful if the route changed in OSM.
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
    Downloads and returns the geometry of the bus line identified
    by its OpenStreetMap relation_id.

    - First request: downloads from Overpass API (~2–10 sec) and caches it.
    - Subsequent requests: responds from in-memory cache (<1 ms).

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
        # The relation probably exists but is not a bus route with the
        # expected way/node structure.
        logger.warning("ValueError for relation %d: %s", relation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not build a route for relation_id {relation_id}. "
                   f"Verify that it is a 'route' relation in OSM. "
                   f"Details: {exc}",
        )
    except RuntimeError as exc:
        # Overpass did not respond — network issue or servers are down.
        logger.error("RuntimeError for relation %d: %s", relation_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not contact Overpass API to download relation {relation_id}. "
                   f"Please try again in a few minutes. Details: {exc}",
        )

    return RouteResponse(
        coordinates=result["coordinates"],
        stops=result["stops"],
        relation_id=result["relation_id"],
        from_cache=result["from_cache"],
    )