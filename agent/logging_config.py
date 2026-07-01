"""
Logging setup for the agent process, mirroring backend/logging_config.py.

Kept as a separate copy (rather than importing from backend/) so the agent
can be deployed/run independently of the backend package.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def log_tracing_status() -> None:
    """
    LangSmith tracing activates purely via env vars (LANGSMITH_TRACING,
    LANGSMITH_API_KEY, LANGSMITH_PROJECT) -- LangChain/LangGraph pick them up
    automatically, no code wiring needed. This just logs whether it's on, so
    it's obvious from the logs alone whether a given run was traced.
    """
    if os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1"):
        project = os.getenv("LANGSMITH_PROJECT", "default")
        logger.info("LangSmith tracing ENABLED (project=%r)", project)
    else:
        logger.info("LangSmith tracing disabled (set LANGSMITH_TRACING=true to enable)")


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
