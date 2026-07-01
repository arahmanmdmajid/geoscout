"""
Terminal entry point for testing the GeoScout agent without any UI.

Usage:
    python -m agent.run "Find the best neighborhoods in Lahore to open a
    coffee shop -- near offices, away from existing cafes, walkable"

Requires the backend server to be running (see backend/server.py) and
OPENAI_API_KEY set in .env.
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv

from agent.graph import build_agent_graph, build_initial_messages
from agent.local_tools import generate_candidate_grid
from agent.logging_config import setup_logging
from agent.mcp_client import discover_mcp_tools

load_dotenv()
setup_logging()

logger = logging.getLogger(__name__)


async def run_brief(brief: str) -> str:
    mcp_tools = await discover_mcp_tools()
    tools = mcp_tools + [generate_candidate_grid]

    graph = build_agent_graph(tools)
    initial_state = {"messages": build_initial_messages(brief)}

    final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 30})
    final_message = final_state["messages"][-1]
    return final_message.content


def main() -> None:
    default_brief = (
        "Find the best neighborhoods in Lahore to open a coffee shop -- "
        "near offices, away from existing cafes, walkable."
    )
    brief = " ".join(sys.argv[1:]) or default_brief

    answer = asyncio.run(run_brief(brief))
    print("\n" + "=" * 60)
    print("GEOSCOUT RECOMMENDATION")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
