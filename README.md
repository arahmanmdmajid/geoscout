# GeoScout — Agentic Site-Selection Assistant for Small Businesses

Type a plain-language business brief, get back a ranked, mapped, justified shortlist of neighborhoods.
Agentic • Grounded in real OpenStreetMap data • MCP-native • LangGraph reasoning

[Live Demo](https://geoscout-ivsqdqtpwwyekvf3xubiax.streamlit.app/) · [Backend API](https://geoscout-backend.onrender.com/health) · [Docker Image](https://hub.docker.com/r/arahman1989/geoscout-backend)

---

## 🔰 Badges

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastMCP](https://img.shields.io/badge/Backend-FastMCP%20%2F%20FastAPI-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph%20%2B%20GPT--4o--mini-1C3C3C)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![OpenStreetMap](https://img.shields.io/badge/Data-OpenStreetMap-7EBC6F?logo=openstreetmap&logoColor=white)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Hosted-Render%20%2B%20Streamlit%20Cloud-46E3B7)
![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-1C3C3C)

---

## 📚 Table of Contents
- [Overview](#overview)
- [What Problem Does It Solve?](#what-problem-does-it-solve)
- [How It Solves These Problems](#how-it-solves-these-problems)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Agent Prompt Design](#agent-prompt-design)
- [Project Structure](#project-structure)
- [Installation & Local Setup](#installation--local-setup)
- [Deployment](#deployment)
- [Observability & Logs](#observability--logs)
- [Live Demo](#live-demo)
- [Safety & Limitations](#safety--limitations)

---

## Overview

**GeoScout** takes a plain-language business brief — e.g. *"Find the best neighborhoods in Lahore to open a coffee shop — near offices, away from existing cafés, walkable"* — and returns a ranked shortlist of candidate locations, each with a numeric score, a written justification, and a map marker.

It's built as an **MCP-native agentic system**: a FastMCP backend exposes geospatial tools, and a LangGraph agent (GPT-4o-mini) discovers and calls those tools autonomously to decompose the brief, gather real data, and reason about tradeoffs — rather than following a fixed script.

---

## What Problem Does It Solve?

Picking a location for a small business usually means manually cross-referencing several things at once: where the potential customers are, where the competition already is, and how walkable/accessible an area is — typically done by eyeballing a map or asking around. GeoScout automates that cross-referencing: it geocodes the city, pulls real points of interest from OpenStreetMap for whatever categories matter to the brief, and scores candidate areas by proximity to what you want to be near (e.g. offices, universities) and away from what you don't (e.g. existing competitors) — then explains *why* each recommendation ranks where it does, in plain English.

---

## How It Solves These Problems

The agent decomposes the brief into a city, a set of "attract" categories, and a set of "avoid" categories, then works through a **LangGraph reasoning loop**: it geocodes the city, samples `find_pois` to gauge density (widening the search radius on its own if a category comes back too sparse — genuine re-planning, not a hardcoded retry), generates a grid of candidate coordinates, and calls `score_sites`, which does the actual proximity-weighted scoring server-side using Haversine distance. The LLM never has to transcribe raw geodata between tool calls — it only ever passes `candidates` + `weights`; the backend fetches and computes everything else. The agent then synthesizes the ranked results into a written recommendation, which the UI pairs with a `folium` map and a per-site "why" breakdown.

---

## Features

**Core:** Chat-style plain-language input, ranked candidate table with scores, per-site justification (structured reasons, not just a number), interactive folium map with color-coded markers, agent tool-call trace for transparency.

**Technical:** MCP tool server (FastMCP on FastAPI/Starlette) with 4 tools + a plain `/health` REST route, LangGraph agent as an MCP client (`langchain-mcp-adapters` + `MultiServerMCPClient`), autonomous re-planning on sparse results, disk-cached + rate-limited OpenStreetMap access with automatic Overpass mirror fallback (retried across 2 full rounds, degrading to an empty-but-valid result rather than crashing the tool call if every mirror is down), structured logging throughout (tool calls, results, re-planning decisions, failures), Dockerized backend.

---

## Tech Stack

**Frontend:** Streamlit, `streamlit-folium`, `folium`

**Agent:** LangGraph, `langchain-mcp-adapters`, `langchain-openai` (GPT-4o-mini)

**Backend:** FastMCP (FastAPI/Starlette under the hood), Uvicorn, `geopy` (Haversine distance), `shapely`

**Data sources:** OpenStreetMap Nominatim (geocoding) + Overpass API (points of interest) — both free, no API key required

**Infra:** Docker, Docker Hub, Render (backend), Streamlit Community Cloud (frontend)

---

## Architecture

```
User types a brief in Streamlit chat
        │
        ▼
LangGraph agent (GPT-4o-mini) — discovers tools via MultiServerMCPClient
        │
        ├─▶ geocode(place) ─────────────────► Nominatim
        ├─▶ find_pois(lat, lon, category) ──► Overpass API (rate-limited, cached, mirror fallback)
        ├─▶ generate_candidate_grid(...)  ──► local deterministic helper (not LLM math)
        └─▶ score_sites(candidates, weights) ► fetches its own POIs via find_pois,
                                                scores by weighted Haversine proximity
        │
        ▼
Ranked, justified shortlist ──► Streamlit: results table + reasons expander + folium map
```

All four geospatial tools (`geocode`, `find_pois`, `compute_distance_matrix`, `score_sites`) live on the FastMCP backend and are exposed over the MCP protocol at `/mcp`; a separate plain `GET /health` route lives alongside them as an ordinary FastAPI endpoint.

---

## Agent Prompt Design

The agent (`agent/graph.py`) is a hand-built LangGraph loop — `agent` node ↔ `tools` node — rather than a prebuilt `create_react_agent`, so every decision point is visible and explainable rather than hidden inside a library:

- **The system prompt is a step-by-step process, not a personality description.** It tells the model explicitly: resolve the city first, decide which known categories to attract/avoid, sample `find_pois` at a starting radius, widen if sparse, generate a candidate grid, call `score_sites`, then write a final answer citing real numbers from the tool results. This keeps the model's freedom scoped to *decisions* (which categories, what radius, when to retry) rather than *invention* (numbers, POIs, coordinates it wasn't given).

- **Re-planning is a nudge, not a hardcoded retry.** After every `find_pois` call, `agent/graph.py`'s `_check_replan()` checks the result count and, if it's below a threshold, injects a plain-language note into the conversation ("...only returned 2 results within 1500m. Consider widening the radius to 3000m..."). The *agent* still decides whether and how to act on that — it's prompted behavior, not a `for` loop forcing a retry. This was a deliberate choice: a hardcoded retry can't adapt if the real fix is a different category or city, but a nudge the LLM interprets can.

- **Precise arithmetic is deliberately kept out of the LLM's hands.** `generate_candidate_grid` (a local, non-MCP tool) computes candidate coordinates with plain Python math, not the model. Early design consideration: LLMs are unreliable at exact lat/lon arithmetic, and a silently-wrong coordinate is a hard bug to notice later — so the model only ever decides *when* to call this tool and with what span, never the numbers themselves.

- **`score_sites` intentionally takes only `candidates` + `weights`, nothing else.** An earlier version also required the agent to pass a `pois_by_category` argument — assembled by hand-copying prior `find_pois` results into the next tool call. In production, this broke on a 99-item category: the model's own JSON generation got corrupted mid-transcription, failing tool validation twice before it stumbled onto a workaround. The fix was architectural, not a prompt tweak: `score_sites` now fetches its own POI data server-side (reusing `find_pois`'s existing caching/rate-limiting/fallback pipeline), so the LLM never transcribes bulk geodata between tool calls at all. This is the single biggest reliability lesson from building this agent — if a tool call requires the model to faithfully relay a previous tool's raw output, expect it to fail once the payload gets large.

---

## Project Structure

```
geoscout/
├── backend/                 # FastMCP server — deployed as a Docker container
│   ├── server.py            # MCP tool registration + /health route
│   ├── geo_client.py        # Nominatim/Overpass clients (rate-limited, cached, mirrored)
│   ├── scoring.py           # Haversine distance + weighted proximity scoring
│   ├── cache.py             # flat-file disk cache for OSM queries
│   └── logging_config.py
├── agent/                   # LangGraph agent (MCP client)
│   ├── graph.py             # agent<->tools loop, re-planning, system prompt
│   ├── mcp_client.py        # MultiServerMCPClient wiring
│   ├── local_tools.py       # generate_candidate_grid (deterministic, non-LLM)
│   ├── run.py                # terminal entry point for testing without a UI
│   └── logging_config.py
├── frontend/
│   └── app.py                # Streamlit chat UI + results table + folium map
├── Dockerfile                 # backend-only image
├── requirements.txt            # full stack (backend + agent + frontend)
├── requirements-backend.txt   # backend-only, used by Dockerfile
└── .env.example
```

---

## Installation & Local Setup

1. **Clone:** `git clone https://github.com/arahmanmdmajid/geoscout` and `cd geoscout`

2. **Virtual environment:** `python -m venv .venv` then activate it
   (Windows: `.venv\Scripts\activate`)

3. **Install:** `pip install -r requirements.txt`

4. **API key:** copy `.env.example` to `.env` and set `OPENAI_API_KEY` to your real key. `MCP_SERVER_URL` defaults to `http://127.0.0.1:8000` for local use.

5. **Run the backend** (terminal 1):
   ```
   python -m backend.server
   ```

6. **Run the UI** (terminal 2, from the project root):
   ```
   streamlit run frontend/app.py
   ```

7. Open `http://localhost:8501` and type a brief.

You can also test the backend alone via the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) (`npx @modelcontextprotocol/inspector`, connect to `http://127.0.0.1:8000/mcp`), or exercise the agent without any UI: `python -m agent.run "your brief here"`.

---

## Deployment

**Backend → Docker Hub → Render:**
```
docker build -t geoscout-backend .
docker tag geoscout-backend:latest arahman1989/geoscout-backend:latest
docker push arahman1989/geoscout-backend:latest
```
On [Render](https://render.com): **New + → Web Service → Deploy an existing image from a registry**, point it at `docker.io/arahman1989/geoscout-backend:latest`, set `NOMINATIM_USER_AGENT` as an environment variable. Render injects its own `PORT`, which `backend/server.py` already reads from the environment.

**Frontend → Streamlit Community Cloud:**
Push this repo to GitHub, then on [share.streamlit.io](https://share.streamlit.io): **New app**, point it at `frontend/app.py`, and add these secrets:
```toml
OPENAI_API_KEY = "sk-..."
MCP_SERVER_URL = "https://your-backend.onrender.com"
NOMINATIM_USER_AGENT = "GeoScout/0.1 (contact: you@example.com)"
```
`frontend/app.py` bridges Streamlit secrets into `os.environ` at startup, so the same `os.getenv()`-based code works identically whether run locally (via `.env`) or deployed.

---

## Observability & Logs

Every layer logs through Python's `logging` module — `logging.info` for normal flow, `logging.error` for failures, and API keys/secrets are never logged.

- **Backend:** each MCP tool call and result (summarized, not the full raw payload), plus Overpass/Nominatim requests and cache hits. Locally, this prints to the terminal running `python -m backend.server`; on Render, it's in the service's **Logs** tab.
- **Agent:** every tool the LLM decides to call, its arguments, the result summary, re-planning decisions (e.g. "find_pois only returned 2 results, widening radius to 3000m"), and the final recommendation. Locally, this is the terminal running `agent.run` or `streamlit run frontend/app.py`; on Streamlit Community Cloud, it's the **"Manage app" → logs** panel.
- **Frontend:** the same agent trace is also rendered directly in the UI, under the **"Agent tool-call trace (observability)"** expander below the results — no need to open a separate logs panel to see what the agent did.

**Tracing (LangSmith):** beyond logs, every agent run is traced end-to-end in [LangSmith](https://smith.langchain.com) — the full LLM reasoning steps, tool calls, and timings for each run, viewable as a structured trace tree rather than a flat log. It activates purely through environment variables (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`) — LangChain/LangGraph auto-instrument once they're set, no code changes needed. A startup log line (`LangSmith tracing ENABLED (project=...)` or `disabled`) confirms which mode a given run is in. Set these in `.env` locally, or in the Streamlit Cloud app's Secrets for the deployed version.

---

## Live Demo

👉 [https://geoscout-ivsqdqtpwwyekvf3xubiax.streamlit.app/](https://geoscout-ivsqdqtpwwyekvf3xubiax.streamlit.app/)

**Demo video:** [Google Drive](https://drive.google.com/drive/folders/1NMcTzEZJhkBl2n1Pnn_MoLwxSoxPVk4r?usp=drive_link)

---

## Safety & Limitations

GeoScout is a research/demo tool for exploring site-selection tradeoffs, not a substitute for professional market research, a real estate agent, or due diligence before signing a lease. Recommendations are only as complete as OpenStreetMap's coverage for a given city — POI density on OSM varies significantly by region, so sparse data in an area doesn't necessarily mean there's genuinely nothing there. Scores are a simple weighted-proximity heuristic (Haversine distance to nearest POI per category), not a full economic or foot-traffic model. The public Nominatim/Overpass endpoints are free, rate-limited, and have no uptime guarantee, which can occasionally slow down or degrade a request — `find_pois` retries across mirrors and degrades to zero results for a category rather than failing the whole run if every mirror is genuinely down, but that does mean a category can occasionally under-report during an outage rather than error loudly. Always verify a specific recommendation on the ground before acting on it.
