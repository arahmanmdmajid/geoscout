"""
GeoScout MCP backend.

This is a FastMCP server (FastMCP wraps FastAPI/Starlette under the hood).
It exposes MCP "tools" that the LangGraph agent will call — geocode and
find_pois for now, compute_distance_matrix and score_sites coming next.

It also exposes ONE plain FastAPI-style route, GET /health, added via
FastMCP's custom_route() so it's clear this is a normal HTTP endpoint,
separate from the MCP protocol routes the tools use.

Run it directly:
    python -m backend.server
"""

import logging
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.logging_config import setup_logging
from backend import geo_client
from backend import scoring

load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)

mcp = FastMCP(name="GeoScout")


@mcp.tool
def geocode(place: str) -> dict:
    """Resolve a place name (e.g. 'Lahore, Pakistan') to lat/lon via OpenStreetMap Nominatim."""
    logger.info("tool call: geocode(place=%r)", place)
    try:
        result = geo_client.geocode(place)
        logger.info("tool result: geocode -> %s (%.5f, %.5f)", result["display_name"], result["lat"], result["lon"])
        return result
    except Exception:
        logger.error("tool failed: geocode(place=%r)", place, exc_info=True)
        raise


@mcp.tool
def find_pois(lat: float, lon: float, category: str, radius: int = 1000) -> list[dict]:
    """
    Find points of interest of a given category within `radius` meters of
    (lat, lon) via OpenStreetMap Overpass. Known categories:
    cafe, coffee_shop, restaurant, office, coworking, bank, university,
    school, park, bus_stop, subway_station, mall, supermarket, retail,
    residential.
    """
    logger.info(
        "tool call: find_pois(lat=%.5f, lon=%.5f, category=%r, radius=%d)",
        lat, lon, category, radius,
    )
    try:
        results = geo_client.find_pois(lat, lon, category, radius)
        logger.info("tool result: find_pois -> %d POIs found", len(results))
        return results
    except Exception:
        logger.error(
            "tool failed: find_pois(lat=%.5f, lon=%.5f, category=%r, radius=%d)",
            lat, lon, category, radius, exc_info=True,
        )
        raise


@mcp.tool
def compute_distance_matrix(points: list[dict]) -> dict:
    """
    Compute pairwise Haversine distances (meters) between a list of points.
    Each point must be {"id": str, "lat": float, "lon": float} ("id" optional).
    Returns {"ids": [...], "distances_m": [[...], ...]}.
    """
    logger.info("tool call: compute_distance_matrix(points=%d)", len(points))
    try:
        result = scoring.compute_distance_matrix(points)
        logger.info("tool result: compute_distance_matrix -> %dx%d matrix", len(points), len(points))
        return result
    except Exception:
        logger.error("tool failed: compute_distance_matrix(points=%d)", len(points), exc_info=True)
        raise


@mcp.tool
def score_sites(
    candidates: list[dict],
    weights: dict[str, float],
    radius_cap: int = 2000,
) -> list[dict]:
    """
    Score and rank candidate sites by proximity to categories of interest.
    Fetches the relevant POIs itself (via find_pois) -- you only need to
    supply candidates and weights, not raw POI data.

    candidates: [{"id": str, "lat": float, "lon": float}, ...]
    weights: {"office": 1.0, "cafe": -1.0}
        Positive weight = candidate should be CLOSE to this category.
        Negative weight = candidate should be FAR from this category
        (e.g. use a negative weight on an existing-competitor category).
        Category names must be known find_pois categories.
    radius_cap: meters beyond which a category's POIs stop influencing score.

    Returns candidates sorted by descending score, each with "score",
    human-readable "reasons", and per-category "components" for transparency.
    """
    logger.info(
        "tool call: score_sites(candidates=%d, weights=%s, radius_cap=%d)",
        len(candidates), weights, radius_cap,
    )
    try:
        result = scoring.score_sites(candidates, weights, radius_cap)
        logger.info(
            "tool result: score_sites -> %d ranked candidates, top score=%.4f",
            len(result), result[0]["score"] if result else 0.0,
        )
        return result
    except Exception:
        logger.error(
            "tool failed: score_sites(candidates=%d, weights=%s)",
            len(candidates), weights, exc_info=True,
        )
        raise


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Plain FastAPI-style health check, separate from the MCP tool routes."""
    return JSONResponse({"status": "ok", "service": "geoscout-backend"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting GeoScout MCP backend on port %d", port)
    mcp.run(transport="http", host="0.0.0.0", port=port)
