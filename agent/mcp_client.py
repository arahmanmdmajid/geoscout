"""
Wires up the LangGraph agent as an MCP client of the GeoScout backend.

MultiServerMCPClient (from langchain-mcp-adapters) connects to one or more
MCP servers and exposes their tools as LangChain-compatible tools — this is
what lets the LLM "discover" geocode/find_pois/compute_distance_matrix/
score_sites at runtime instead of us hardcoding tool schemas by hand.
"""

import logging
import os

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


def build_mcp_client() -> MultiServerMCPClient:
    server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
    mcp_endpoint = f"{server_url.rstrip('/')}/mcp"

    logger.info("connecting to GeoScout MCP server at %s", mcp_endpoint)
    return MultiServerMCPClient({
        "geoscout": {
            "transport": "streamable_http",
            "url": mcp_endpoint,
        }
    })


async def discover_mcp_tools() -> list:
    client = build_mcp_client()
    tools = await client.get_tools()
    logger.info("discovered %d MCP tools: %s", len(tools), [t.name for t in tools])
    return tools
