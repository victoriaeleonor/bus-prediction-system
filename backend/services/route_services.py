"""
route_service.py
================
Downloads the geometry of a bus route from OpenStreetMap (Overpass API),
reconstructs the ordered polyline from the relation ways,
and caches the result in memory to avoid repeating the HTTP request on every call.

Data source
-----------
Overpass API — https://overpass-api.de
The frontend references relation 3983243 as an example:
  https://www.openstreetmap.org/relation/3983243

Query used (ordered skeleton):
  [out:json];relation({relation_id});way(r);node(w);out skel qt;

The response contains elements of type "relation", "way", and "node".
Using these three types, the route can be reconstructed in the correct order
by following the relation members.
"""

import logging
import math
from typing import Optional

import httpx  # async HTTP — replaces requests for FastAPI/asyncio usage

logger = logging.getLogger(__name__)

# ── Overpass servers (tested in order) ─────────────────────────────
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Timeout per server (seconds). Overpass can be slow during peak hours.
OVERPASS_TIMEOUT = 30

# Maximum discontinuity (lat/lon degrees) between the end of one way and the
# start of the next to consider them connected. Above this threshold,
# the segment is discarded (gaps in the OSM relation).
MAX_GAP_DEGREES = 0.015

# Subsampling: take 1 out of every N points from the final route to avoid
# overloading the map with thousands of unnecessary coordinates.
ROUTE_SUBSAMPLE = 2


# ── in-memory cache ──────────────────────────────────────────────────────
# Key: relation_id (int)
# Value: dict with "coordinates" and "stops" ([lat, lon] lists)
#
# In production, this could be replaced with Redis, but for the current
# project scope (one or a few active routes), memory is sufficient.
_route_cache: dict[int, dict] = {}


# ── geometric helpers ───────────────────────────────────────────────────

def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points (Haversine formula)."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _reconstruct_polyline(
    elements: list[dict],
) -> list[tuple[float, float]]:
    """
    Reconstructs the ordered polyline from Overpass elements.

    Overpass with 'out skel qt' returns nodes ordered within each way,
    but the ways inside the relation may be in any order and
    some may be reversed. This function:

    1. Builds a node map (id → lat/lon).
    2. Builds a way map (id → list of node_ids).
    3. Follows the order of relation members, detecting whether each way
       should be reversed to connect with the previous segment.
    4. Discards segments with a gap larger than MAX_GAP_DEGREES.
    5. Applies subsampling at the end to reduce data volume.

    Returns a list of (lat, lon).
    """
    node_map: dict[int, tuple[float, float]] = {
        el["id"]: (el["lat"], el["lon"])
        for el in elements
        if el["type"] == "node"
    }
    way_map: dict[int, list[int]] = {
        el["id"]: el["nodes"]
        for el in elements
        if el["type"] == "way"
    }

    relation = next(
        (el for el in elements if el["type"] == "relation"),
        None,
    )
    if relation is None:
        raise ValueError("The Overpass response does not contain any relation.")

    route: list[tuple[float, float]] = []
    prev_last: Optional[tuple[float, float]] = None

    for member in relation["members"]:
        if member["type"] != "way":
            continue
        way_id = member["ref"]
        if way_id not in way_map:
            continue

        coords = [node_map[n] for n in way_map[way_id] if n in node_map]
        if not coords:
            continue

        if prev_last is not None:
            # Detect whether the way is reversed relative to the previous segment
            d_fwd = abs(coords[0][0] - prev_last[0]) + abs(coords[0][1] - prev_last[1])
            d_rev = abs(coords[-1][0] - prev_last[0]) + abs(coords[-1][1] - prev_last[1])

            if d_rev < d_fwd:
                coords = coords[::-1]

            # Discard if the gap is too large (incomplete OSM relation)
            if min(d_fwd, d_rev) > MAX_GAP_DEGREES:
                logger.debug(
                    "Way %s discarded: gap of %.4f° with the previous segment.",
                    way_id, min(d_fwd, d_rev),
                )
                continue

        route.extend(coords)
        prev_last = route[-1]

    if not route:
        raise ValueError(
            "Could not reconstruct any polyline from the relation."
        )

    return route[::ROUTE_SUBSAMPLE]


def _extract_stops(elements: list[dict]) -> list[tuple[float, float]]:
    """
    Extracts nodes marked as stops within the relation.

    In OSM, bus route stops appear in the relation members with
    role "stop" or "stop_entry_only" / "stop_exit_only",
    and are nodes tagged with public_transport=stop_position
    or highway=bus_stop.

    If the relation does not contain tagged stops (as happens
    with some simple relations), an empty list is returned:
    the frontend will use the polyline coordinates as a visual reference.
    """
    relation = next(
        (el for el in elements if el["type"] == "relation"),
        None,
    )
    if relation is None:
        return []

    stop_ids: list[int] = [
        m["ref"]
        for m in relation["members"]
        if m["type"] == "node" and "stop" in m.get("role", "")
    ]

    node_map: dict[int, tuple[float, float]] = {
        el["id"]: (el["lat"], el["lon"])
        for el in elements
        if el["type"] == "node"
    }

    stops = [node_map[sid] for sid in stop_ids if sid in node_map]
    logger.info("Stops found in relation: %d", len(stops))
    return stops


# ── Overpass query ─────────────────────────────────────────────────────────

async def _fetch_from_overpass(relation_id: int) -> list[dict]:
    """
    Calls the Overpass API and returns the JSON elements list.

    Tries the servers in OVERPASS_SERVERS in order;
    if all fail, raises RuntimeError.

    Uses httpx.AsyncClient to avoid blocking the FastAPI event loop.
    """
    # Skeleton query with all nodes from the relation ways,
    # including member nodes (stops) from the relation itself.
    query = (
        f"[out:json];"
        f"relation({relation_id});"
        f"out body;"          # fetch relation with members and tags
        f">;"                 # expand: all referenced ways and nodes
        f"out skel qt;"       # ordered geometry (qt = quadtile quicksort)
    )

    last_error: Exception = RuntimeError("No server was attempted.")

    async with httpx.AsyncClient(verify=False, timeout=OVERPASS_TIMEOUT) as client:
        for server in OVERPASS_SERVERS:
            logger.info("Querying Overpass: %s (relation %d)", server, relation_id)
            try:
                response = await client.post(
                    server,
                    data={"data": query},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                data = response.json()
                elements = data.get("elements", [])
                if not elements:
                    raise ValueError("Overpass returned an empty elements list.")
                logger.info(
                    "Overpass OK — %d elements (relation %d)",
                    len(elements), relation_id,
                )
                return elements
            except Exception as exc:
                logger.warning("Server %s failed: %s", server, exc)
                last_error = exc

    raise RuntimeError(
        f"All Overpass servers failed for relation {relation_id}. "
        f"Last error: {last_error}"
    )


# ── public function ───────────────────────────────────────────────────────

async def get_route(relation_id: int) -> dict:
    """
    Returns the complete geometry of a bus route given its OSM relation_id.

    Response:
    {
        "relation_id": 3983243,
        "coordinates": [[lat, lon], ...],   # complete subsampled polyline
        "stops": [[lat, lon], ...],         # stops (may be empty)
        "from_cache": bool
    }

    The first call downloads from Overpass and caches the result.
    Subsequent calls return the cached result instantly.

    Example usage from a FastAPI router:
        result = await get_route(3983243)
    """
    if relation_id in _route_cache:
        logger.debug("Cache hit for relation_id %d", relation_id)
        return {**_route_cache[relation_id], "from_cache": True}

    logger.info("Cache miss — downloading relation %d from Overpass", relation_id)
    elements = await _fetch_from_overpass(relation_id)

    polyline = _reconstruct_polyline(elements)
    stops = _extract_stops(elements)

    result = {
        "relation_id": relation_id,
        "coordinates": [list(p) for p in polyline],
        "stops": [list(p) for p in stops],
    }

    _route_cache[relation_id] = result
    logger.info(
        "Relation %d cached — %d points, %d stops",
        relation_id, len(polyline), len(stops),
    )
    return {**result, "from_cache": False}


def invalidate_cache(relation_id: Optional[int] = None) -> None:
    """
    Invalidates the cache for a specific relation, or the entire cache if
    relation_id is None. Useful for forcing a re-download without restarting
    the server (for example, if the route changed in OSM).
    """
    if relation_id is None:
        _route_cache.clear()
        logger.info("Route cache completely cleared.")
    else:
        _route_cache.pop(relation_id, None)
        logger.info("Cache invalidated for relation_id %d", relation_id)


def list_cached_routes() -> list[int]:
    """Returns the currently cached relation_ids. Useful for /health."""
    return list(_route_cache.keys())