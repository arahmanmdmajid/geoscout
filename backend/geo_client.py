"""
Thin clients around OpenStreetMap's free, no-API-key data sources:

- Nominatim (https://nominatim.org) for geocoding a place name -> lat/lon.
- Overpass API (https://overpass-api.de) for querying POIs (points of
  interest) near a location by OSM tag (e.g. amenity=cafe, office=*).

Both are shared public services with usage policies we must respect:
  - Nominatim requires a descriptive User-Agent identifying the app/contact,
    and a max of 1 request/second.
  - Overpass asks that clients not hammer it either; we apply the same
    1 req/sec limiter to be safe.

To avoid tripping these limits while developing (repeatedly re-running the
same query), every call is cached on disk first — see cache.py.
"""

import logging
import os
import threading
import time

import requests

from backend.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# The public Overpass instance gets overloaded and returns 504s fairly often.
# We try it first, then fall back to other public mirrors that serve the same
# API, so a single busy server doesn't make find_pois unreliable.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "GeoScout/0.1 (no-contact-set)")

MIN_REQUEST_INTERVAL_SECONDS = 1.0

# Maps a plain-language category (what the agent/user will ask for) to OSM
# tag(s) to search for. "*" means "any value for this key".
CATEGORY_TAGS: dict[str, list[tuple[str, str]]] = {
    "cafe": [("amenity", "cafe")],
    "coffee_shop": [("amenity", "cafe")],
    "restaurant": [("amenity", "restaurant")],
    "bakery": [("shop", "bakery")],
    "fast_food": [("amenity", "fast_food")],
    "office": [("office", "*")],
    "coworking": [("office", "coworking")],
    "bank": [("amenity", "bank")],
    "university": [("amenity", "university")],
    "school": [("amenity", "school")],
    "park": [("leisure", "park")],
    "bus_stop": [("highway", "bus_stop")],
    "subway_station": [("railway", "station")],
    "mall": [("shop", "mall")],
    "supermarket": [("shop", "supermarket")],
    "retail": [("shop", "*")],
    "residential": [("landuse", "residential")],
}


class _RateLimiter:
    """Ensures at least 1 second between real (non-cached) outbound requests."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()


_rate_limiter = _RateLimiter(MIN_REQUEST_INTERVAL_SECONDS)


def geocode(place: str) -> dict:
    """
    Resolve a place name (e.g. "Lahore, Pakistan") to coordinates + metadata
    via Nominatim. Returns the top match.
    """
    cache_key = place.strip().lower()
    cached = cache_get("geocode", cache_key)
    if cached is not None:
        logger.info("geocode cache hit for place=%r", place)
        return cached

    logger.info("geocode: calling Nominatim for place=%r", place)
    _rate_limiter.wait()

    response = requests.get(
        NOMINATIM_URL,
        params={"q": place, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()

    if not results:
        logger.error("geocode: no results found for place=%r", place)
        raise ValueError(f"No geocoding results found for '{place}'")

    top = results[0]
    result = {
        "place": place,
        "lat": float(top["lat"]),
        "lon": float(top["lon"]),
        "display_name": top.get("display_name", place),
        "bounding_box": top.get("boundingbox"),
    }
    cache_set("geocode", cache_key, result)
    logger.info(
        "geocode: resolved place=%r -> lat=%.5f lon=%.5f",
        place, result["lat"], result["lon"],
    )
    return result


def _build_overpass_query(lat: float, lon: float, radius: int, tags: list[tuple[str, str]]) -> str:
    clauses = []
    for key, value in tags:
        tag_filter = f'["{key}"]' if value == "*" else f'["{key}"="{value}"]'
        for element in ("node", "way", "relation"):
            clauses.append(f'{element}{tag_filter}(around:{radius},{lat},{lon});')
    body = "\n  ".join(clauses)
    return f"""
[out:json][timeout:25];
(
  {body}
);
out center;
""".strip()


def _query_overpass_with_fallback(query: str) -> list[dict]:
    """
    POST the given Overpass QL query, trying each known public mirror in turn.
    The shared public instances are free but have no SLA and often return
    504s under load, so we retry across mirrors rather than failing outright.
    """
    last_error: Exception | None = None
    for url in OVERPASS_URLS:
        _rate_limiter.wait()
        try:
            response = requests.post(
                url,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("elements", [])
        except requests.exceptions.RequestException as exc:
            logger.error("find_pois: Overpass mirror %s failed (%s), trying next", url, exc)
            last_error = exc

    logger.error("find_pois: all Overpass mirrors failed")
    raise last_error


def find_pois(lat: float, lon: float, category: str, radius: int = 1000) -> list[dict]:
    """
    Find points of interest of a given category within `radius` meters of
    (lat, lon), via the Overpass API.

    `category` must be one of CATEGORY_TAGS' keys (e.g. "cafe", "office").
    """
    category_key = category.strip().lower()
    tags = CATEGORY_TAGS.get(category_key)
    if tags is None:
        logger.error("find_pois: unknown category=%r", category)
        raise ValueError(
            f"Unknown category '{category}'. Known categories: {sorted(CATEGORY_TAGS)}"
        )

    cache_key = f"{lat:.5f}:{lon:.5f}:{category_key}:{radius}"
    cached = cache_get("find_pois", cache_key)
    if cached is not None:
        logger.info(
            "find_pois cache hit for category=%r radius=%dm -> %d results",
            category_key, radius, len(cached),
        )
        return cached

    query = _build_overpass_query(lat, lon, radius, tags)
    logger.info(
        "find_pois: calling Overpass for category=%r lat=%.5f lon=%.5f radius=%dm",
        category_key, lat, lon, radius,
    )

    elements = _query_overpass_with_fallback(query)

    pois = []
    for el in elements:
        if el["type"] == "node":
            poi_lat, poi_lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            poi_lat, poi_lon = center.get("lat"), center.get("lon")
        if poi_lat is None or poi_lon is None:
            continue
        tags_dict = el.get("tags", {})
        pois.append({
            "id": el["id"],
            "name": tags_dict.get("name", "Unnamed"),
            "lat": poi_lat,
            "lon": poi_lon,
            "tags": tags_dict,
        })

    cache_set("find_pois", cache_key, pois)
    logger.info(
        "find_pois: category=%r radius=%dm -> %d results",
        category_key, radius, len(pois),
    )
    return pois
