"""
Tiny on-disk JSON cache for Nominatim/Overpass responses.

Why this exists: Nominatim and Overpass are free public services with strict
usage policies (Nominatim: max 1 request/sec, custom User-Agent required).
While developing, we call the same geocode/POI queries over and over — this
cache avoids re-hitting the real API for a query we already made, so we don't
get rate-limited or (in the worst case) IP-banned during testing.

Not meant as a production cache — just a flat JSON-per-key file store under
.cache/, which is already gitignored.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _cache_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / namespace / f"{digest}.json"


def cache_get(namespace: str, key: str) -> Any | None:
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.error("cache read failed for %s/%s, ignoring stale entry", namespace, key)
        return None


def cache_set(namespace: str, key: str, value: Any) -> None:
    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f)
