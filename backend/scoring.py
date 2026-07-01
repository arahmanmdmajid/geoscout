"""
Distance and scoring logic for GeoScout.

Two building blocks:

- compute_distance_matrix: generic pairwise Haversine distances (meters)
  between a list of points. Useful for e.g. checking how spread out the
  shortlisted candidates are.

- score_sites: given candidate locations and, for each category the user
  cares about (e.g. "office", "cafe"), the POIs of that category nearby,
  score each candidate by proximity to each category, weighted by how much
  the user wants to be near (positive weight) or far from (negative weight)
  that category.

All distances use the Haversine formula (via geopy.distance.great_circle),
which treats the Earth as a sphere — accurate enough for neighborhood-level
comparisons and simple to reason about.
"""

import logging

from geopy.distance import great_circle

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


def score_sites(
    candidates: list[dict],
    pois_by_category: dict[str, list[dict]],
    weights: dict[str, float],
    radius_cap: int = DEFAULT_RADIUS_CAP_METERS,
) -> list[dict]:
    """
    candidates: list of {"id": str, "lat": float, "lon": float}
    pois_by_category: {"office": [{"lat":.., "lon":..}, ...], "cafe": [...]}
    weights: {"office": 1.0, "cafe": -1.0}
        Positive weight = candidate should be CLOSE to this category (e.g.
        offices, foot traffic). Negative weight = candidate should be FAR
        from this category (e.g. existing competitor cafes).
    radius_cap: distance in meters at which proximity influence saturates to
        zero — a POI further than this away no longer helps or hurts the
        score for that category.

    Returns candidates sorted by descending score, each annotated with:
        - score: float
        - reasons: list[str] human-readable justification lines
        - components: per-category {nearest_m, count_within_radius, contribution}
    """
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
