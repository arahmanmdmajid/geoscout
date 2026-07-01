"""
Distance and scoring logic for GeoScout.

Two building blocks:

- compute_distance_matrix: generic pairwise Haversine distances (meters)
  between a list of points. Useful for e.g. checking how spread out the
  shortlisted candidates are.

- score_sites: given candidate locations and a set of category weights (e.g.
  {"office": 1.0, "cafe": -1.0}), fetches the relevant POIs itself (via
  find_pois) and scores each candidate by proximity, weighted by how much the
  user wants to be near (positive weight) or far from (negative weight) that
  category.

  Deliberately, the caller (the LLM agent) never has to pass raw POI lists
  in — only candidates + weights, matching the tool's intended interface.
  Earlier this required the agent to copy potentially 100+ POI objects from
  a prior find_pois result into this call's arguments, which is exactly the
  kind of large, mechanical transcription task LLMs get wrong (this broke in
  practice on a 99-POI category, producing malformed tool-call JSON
  mid-array). Fetching POIs server-side avoids that failure mode entirely.

All distances use the Haversine formula (via geopy.distance.great_circle),
which treats the Earth as a sphere — accurate enough for neighborhood-level
comparisons and simple to reason about.
"""

import logging

from geopy.distance import great_circle

from backend.geo_client import find_pois

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_CAP_METERS = 2000


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (Haversine) distance between two points, in meters."""
    return great_circle((lat1, lon1), (lat2, lon2)).meters


def compute_distance_matrix(points: list[dict]) -> dict:
    """
    points: list of {"id": str, "lat": float, "lon": float}
    Returns {"ids": [...], "distances_m": [[...], ...]} — a symmetric matrix
    where distances_m[i][j] is the Haversine distance in meters between
    points[i] and points[j].
    """
    ids = [p.get("id", str(i)) for i, p in enumerate(points)]
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance_m(
                points[i]["lat"], points[i]["lon"], points[j]["lat"], points[j]["lon"]
            )
            matrix[i][j] = dist
            matrix[j][i] = dist

    return {"ids": ids, "distances_m": matrix}


def _nearest_distance_m(lat: float, lon: float, pois: list[dict]) -> float | None:
    if not pois:
        return None
    return min(haversine_distance_m(lat, lon, poi["lat"], poi["lon"]) for poi in pois)


def _fetch_pois_by_category(candidates: list[dict], categories: list[str], radius_cap: int) -> dict[str, list[dict]]:
    """
    Fetch POIs for each category once, centered on the candidates' centroid,
    with a search radius wide enough to cover every candidate plus radius_cap
    beyond the farthest one — so no candidate misses POIs that are within
    radius_cap of it specifically.
    """
    center_lat = sum(c["lat"] for c in candidates) / len(candidates)
    center_lon = sum(c["lon"] for c in candidates) / len(candidates)
    max_dist_from_center = max(
        haversine_distance_m(center_lat, center_lon, c["lat"], c["lon"]) for c in candidates
    )
    search_radius = int(max_dist_from_center + radius_cap)

    pois_by_category = {}
    for category in categories:
        pois = find_pois(center_lat, center_lon, category, search_radius)
        pois_by_category[category] = pois
        logger.info(
            "score_sites: fetched %d '%s' POIs within %dm of candidate centroid",
            len(pois), category, search_radius,
        )
    return pois_by_category


def score_sites(
    candidates: list[dict],
    weights: dict[str, float],
    radius_cap: int = DEFAULT_RADIUS_CAP_METERS,
) -> list[dict]:
    """
    candidates: list of {"id": str, "lat": float, "lon": float}
    weights: {"office": 1.0, "cafe": -1.0}
        Positive weight = candidate should be CLOSE to this category (e.g.
        offices, foot traffic). Negative weight = candidate should be FAR
        from this category (e.g. existing competitor cafes). Category names
        must be known find_pois categories (e.g. cafe, office, university).
    radius_cap: distance in meters at which proximity influence saturates to
        zero — a POI further than this away no longer helps or hurts the
        score for that category.

    Returns candidates sorted by descending score, each annotated with:
        - score: float
        - reasons: list[str] human-readable justification lines
        - components: per-category {nearest_m, count_within_radius, contribution}
    """
    if not candidates:
        return []

    pois_by_category = _fetch_pois_by_category(candidates, list(weights.keys()), radius_cap)

    scored = []

    for candidate in candidates:
        lat, lon = candidate["lat"], candidate["lon"]
        total_score = 0.0
        components = {}
        reasons = []

        for category, weight in weights.items():
            pois = pois_by_category.get(category, [])
            nearest_m = _nearest_distance_m(lat, lon, pois)
            count_within_radius = sum(
                1 for poi in pois
                if haversine_distance_m(lat, lon, poi["lat"], poi["lon"]) <= radius_cap
            )

            if nearest_m is None:
                proximity = 0.0
            else:
                proximity = max(0.0, 1.0 - nearest_m / radius_cap)

            contribution = weight * proximity
            total_score += contribution
            components[category] = {
                "nearest_m": nearest_m,
                "count_within_radius": count_within_radius,
                "contribution": contribution,
            }

            direction = "near" if weight >= 0 else "away from"
            if nearest_m is None:
                reasons.append(f"No {category} found nearby (wanted {direction} {category}).")
            else:
                reasons.append(
                    f"{count_within_radius} {category}(s) within {radius_cap}m, "
                    f"nearest {nearest_m:.0f}m away (wanted {direction} {category})."
                )

        scored.append({
            "id": candidate.get("id"),
            "lat": lat,
            "lon": lon,
            "score": round(total_score, 4),
            "reasons": reasons,
            "components": components,
        })

    scored.sort(key=lambda c: c["score"], reverse=True)
    logger.info(
        "score_sites: scored %d candidates, top score=%.4f (id=%s)",
        len(scored), scored[0]["score"] if scored else 0.0, scored[0]["id"] if scored else None,
    )
    return scored
