"""
Shared logging setup for the backend.

Every module in the project calls setup_logging() once at import time and then
uses logging.getLogger(__name__) to log. Centralizing the format here means the
whole app (backend tools, health route, and later the agent) produces
consistent, readable log lines like:

    2026-07-01 20:10:03 INFO     backend.tools.geocode: geocoding place='Lahore, Pakistan'

We deliberately never log request/response bodies in full (only summaries),
and never log anything read from environment variables (API keys, etc).
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. re-imported) — don't add duplicate handlers.
        return

    root.setLevel(level)

    # On Windows, stdout can default to a legacy codepage (e.g. cp1252) that
    # can't encode non-ASCII text (Urdu place names, etc). Force UTF-8 with a
    # safe fallback so a foreign-language place name never crashes logging.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
