"""
Local (non-MCP) helper tool for the agent.

The backend's MCP tools (geocode, find_pois, compute_distance_matrix,
score_sites) don't include a way to generate "candidate site" coordinates —
score_sites needs a list of candidate lat/lon points to evaluate, but nothing
produces those points for it.

We deliberately do NOT ask the LLM to invent lat/lon grid coordinates itself
— language models are unreliable at precise arithmetic, and a slightly-off
coordinate is a silent, hard-to-spot error. Instead this is a plain
deterministic Python function exposed to the agent as a tool, so the LLM
decides WHEN to generate candidates and with what spacing, but the actual
numbers are computed exactly.
"""

import logging
import math

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

METERS_PER_DEGREE_LAT = 111_320


@tool
def generate_candidate_grid(center_lat: float, center_lon: float, span_m: int, grid_size: int = 3) -> list[dict]:
    """
    Generate a grid of candidate site coordinates covering a square area of
    side `span_m` meters, centered on (center_lat, center_lon).

    Use this AFTER geocoding a city to produce candidate points to evaluate
    with score_sites. `grid_size` is the number of points per side (e.g. 3
    gives a 3x3 = 9 candidate grid). Returns a list of {"id", "lat", "lon"}.
    """
    logger.info(
        "local tool call: generate_candidate_grid(center=(%.5f, %.5f), span_m=%d, grid_size=%d)",
        center_lat, center_lon, span_m, grid_size,
    )

    half_span = span_m / 2
    dlat = half_span / METERS_PER_DEGREE_LAT
    dlon = half_span / (METERS_PER_DEGREE_LAT * math.cos(math.radians(center_lat)))

    if grid_size == 1:
        offsets = [0.0]
    else:
        offsets = [-1.0 + 2.0 * i / (grid_size - 1) for i in range(grid_size)]

    candidates = []
    idx = 0
    for lat_offset in offsets:
        for lon_offset in offsets:
            candidates.append({
                "id": f"grid_{idx}",
                "lat": round(center_lat + lat_offset * dlat, 6),
                "lon": round(center_lon + lon_offset * dlon, 6),
            })
            idx += 1

    logger.info("local tool result: generate_candidate_grid -> %d candidates", len(candidates))
    return candidates
