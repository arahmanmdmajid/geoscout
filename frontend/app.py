"""
GeoScout Streamlit front-end.

Talks to the LangGraph agent (agent/graph.py), which in turn talks to the
FastMCP backend over HTTP. This file has three parts:

1. A chat input where the user types a plain-language business brief.
2. A results area: a table of ranked candidate sites with scores + reasons.
3. A folium map with a marker per candidate, colored green/red by score.

The agent graph itself is async (it awaits MCP tool calls), but Streamlit's
script model is synchronous, so we use asyncio.run() to drive it each time.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# `streamlit run frontend/app.py` (as opposed to `python -m streamlit run ...`)
# puts this file's own directory on sys.path instead of the project root, so
# the sibling `agent` package wouldn't otherwise be importable. Adding the
# project root explicitly makes this work regardless of how it's launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import folium
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from agent.graph import _summarize_args, _summarize_result, build_agent_graph, build_initial_messages
from agent.local_tools import generate_candidate_grid
from agent.logging_config import setup_logging
from agent.mcp_client import discover_mcp_tools

load_dotenv(PROJECT_ROOT / ".env")
setup_logging()

# On Streamlit Community Cloud, secrets set in the app dashboard are exposed
# via st.secrets, not as environment variables. Locally, .env + load_dotenv()
# already populates os.environ. Bridging st.secrets into os.environ here lets
# the rest of the codebase (agent/, backend/) keep using plain os.getenv()
# either way, without knowing which environment it's running in.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except st.errors.StreamlitSecretNotFoundError:
    pass  # no secrets.toml locally — that's fine, .env already covered it

st.set_page_config(page_title="GeoScout", page_icon="🗺️", layout="wide")


@st.cache_resource
def get_agent_graph():
    """Discover MCP tools once and compile the graph, cached across reruns."""
    mcp_tools = asyncio.run(discover_mcp_tools())
    tools = mcp_tools + [generate_candidate_grid]
    return build_agent_graph(tools)


def extract_ranked_sites(messages: list) -> list[dict] | None:
    """Pull the structured score_sites output out of the message history."""
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "score_sites":
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def extract_tool_trace(messages: list) -> list[str]:
    """Build a readable list of "tool_name(args) -> summary" lines for display."""
    trace = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                trace.append(f"CALL  {call['name']}({_summarize_args(call['args'])})")
        elif getattr(msg, "type", None) == "tool":
            try:
                content = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                content = msg.content
            trace.append(f"  ->  {msg.name}: {_summarize_result(msg.name, content)}")
    return trace


st.title("🗺️ GeoScout")
st.caption("Describe a business brief in plain language — GeoScout geocodes the city, "
           "pulls real OpenStreetMap data, and returns a ranked, justified shortlist.")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_ranked_sites" not in st.session_state:
    st.session_state.last_ranked_sites = None
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

brief = st.chat_input(
    "e.g. Find the best neighborhoods in Lahore to open a coffee shop — "
    "near offices, away from existing cafes, walkable"
)

if brief:
    st.session_state.history.append(("user", brief))
    with st.chat_message("user"):
        st.markdown(brief)

    with st.chat_message("assistant"):
        with st.spinner("Geocoding, pulling POIs, and scoring candidate sites..."):
            graph = get_agent_graph()
            initial_state = {"messages": build_initial_messages(brief)}
            final_state = asyncio.run(graph.ainvoke(initial_state, config={"recursion_limit": 30}))
            answer = final_state["messages"][-1].content
            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
    st.session_state.last_ranked_sites = extract_ranked_sites(final_state["messages"])
    st.session_state.last_trace = extract_tool_trace(final_state["messages"])

if st.session_state.last_ranked_sites:
    sites = st.session_state.last_ranked_sites

    st.subheader("Ranked candidates")
    df = pd.DataFrame(sites)[["id", "lat", "lon", "score"]]
    st.dataframe(df, width="stretch", hide_index=True)

    with st.expander("Why each site scored this way"):
        for site in sites:
            st.markdown(f"**{site['id']}** — score {site['score']}")
            for reason in site.get("reasons", []):
                st.markdown(f"- {reason}")

    st.subheader("Map")
    center_lat = sum(s["lat"] for s in sites) / len(sites)
    center_lon = sum(s["lon"] for s in sites) / len(sites)
    site_map = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    for site in sites:
        popup_html = f"<b>{site['id']}</b><br>Score: {site['score']}<br>" + "<br>".join(site.get("reasons", []))
        folium.Marker(
            location=[site["lat"], site["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{site['id']} (score {site['score']})",
            icon=folium.Icon(color="green" if site["score"] >= 0 else "red"),
        ).add_to(site_map)
    st_folium(site_map, width=1100, height=500)

if st.session_state.last_trace:
    with st.expander("Agent tool-call trace (observability)"):
        for line in st.session_state.last_trace:
            st.text(line)
