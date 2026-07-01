"""
GeoScout's LangGraph agent.

This is a classic "ReAct-style" LangGraph loop:

    START -> agent -> (has tool calls?) -> tools -> agent -> ... -> END

- "agent" node: the LLM (gpt-4o-mini) looks at the conversation so far and
  either (a) decides to call one or more tools, or (b) writes a final answer.
- "tools" node: actually executes whatever tools the LLM asked for (our 4
  MCP tools + the local generate_candidate_grid helper), logs each call and
  result, and feeds the results back into the conversation as ToolMessages.
- The edge after "tools" always goes back to "agent", so the LLM sees tool
  results and decides its next move — this is the "decompose the brief,
  decide which tools to call" loop the project calls for.

Re-planning (widening the radius) is implemented as a *nudge*: after running
find_pois, if it returned suspiciously few results, we inject an extra
message into the conversation telling the LLM so — and it's the LLM's own
decision (following its system prompt) to act on that nudge by calling
find_pois again with a bigger radius. This keeps the re-planning genuinely
agentic rather than a hardcoded retry loop.
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"

# If find_pois returns fewer than this many results, we nudge the agent to
# retry with a wider radius (as long as it hasn't already gone past the cap).
MIN_POIS_BEFORE_REPLAN = 3
MAX_REPLAN_RADIUS_M = 6000

SYSTEM_PROMPT = """You are GeoScout, a site-selection assistant for small businesses.

You are given a plain-language business brief (e.g. "Find the best neighborhoods
in Lahore to open a coffee shop — near offices, away from existing cafés, walkable").
Your job is to decompose it and use your tools to produce a ranked shortlist of
candidate locations with a written justification for each.

Follow this process:
1. Identify the city/place in the brief and call `geocode` to resolve it.
2. Identify which POI categories the brief wants to be NEAR (e.g. "office" for
   foot traffic from workers) and which it wants to be FAR FROM (e.g. "cafe" to
   avoid existing competition). Only use these known category names when calling
   find_pois: cafe, coffee_shop, restaurant, bakery, fast_food, office, coworking,
   bank, university, school, park, bus_stop, subway_station, mall, supermarket,
   retail, residential.
3. Call `find_pois` once per relevant category, centered on the geocoded
   coordinates, with a reasonable starting radius (e.g. 1000-1500m), to get a
   feel for how many of each exist nearby. If a tool result tells you a
   category returned too few POIs, widen the radius (e.g. double it, up to
   6000m) and call find_pois again for that category — this is expected and
   is how you re-plan.
4. Call `generate_candidate_grid` centered on the geocoded coordinates to get
   a set of candidate site coordinates to evaluate (a span of 3000-5000m and
   grid_size of 3 is a reasonable default).
5. Call `score_sites` with those candidates and a `weights` dict where
   categories to be NEAR get a positive weight (e.g. 1.0) and categories to
   be FAR FROM get a negative weight (e.g. -1.0). You do NOT need to pass POI
   data yourself — score_sites fetches it internally. Just candidates +
   weights.
6. Once you have ranked results, write your FINAL ANSWER as a numbered list of
   the top 3-5 sites. For each: give its id, lat/lon, score, and a short
   (1-2 sentence) natural-language justification built from the "reasons" and
   "components" fields score_sites gave you. Do not call any more tools once
   you've written your final answer.

Always base your justification on the actual tool results — do not invent
numbers or POIs that didn't come back from a tool call.
"""


def _extract_tool_result(result):
    """
    Normalize a tool's raw return value into plain Python data.

    MCP tools (via langchain-mcp-adapters) come back as a list of content
    blocks, e.g. [{"type": "text", "text": "<json string>"}], rather than
    the parsed value directly. Local (non-MCP) tools like
    generate_candidate_grid return plain Python objects already. This
    handles both.
    """
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        combined_text = "".join(block.get("text", "") for block in result if isinstance(block, dict))
        try:
            return json.loads(combined_text)
        except json.JSONDecodeError:
            return combined_text
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def _summarize_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items()) or "{}"


def _summarize_result(name: str, result) -> str:
    """Human-readable one-liner for logs — never the full raw payload."""
    try:
        if name == "geocode":
            return f"{result.get('display_name', '?')} ({result.get('lat')}, {result.get('lon')})"
        if name == "find_pois":
            return f"{len(result)} POIs found"
        if name == "compute_distance_matrix":
            n = len(result.get("ids", []))
            return f"{n}x{n} distance matrix"
        if name == "score_sites":
            top = result[0] if result else None
            return f"{len(result)} ranked candidates, top={top['id'] if top else None} score={top['score'] if top else None}"
        if name == "generate_candidate_grid":
            return f"{len(result)} candidate points"
    except Exception:
        pass
    return str(result)[:200]


def _check_replan(name: str, args: dict, result) -> str | None:
    """Returns a nudge message if this tool result looks like it needs re-planning."""
    if name != "find_pois":
        return None
    if not isinstance(result, list):
        return None
    if len(result) >= MIN_POIS_BEFORE_REPLAN:
        return None

    current_radius = args.get("radius", 1000)
    if current_radius >= MAX_REPLAN_RADIUS_M:
        return None

    new_radius = min(current_radius * 2, MAX_REPLAN_RADIUS_M)
    category = args.get("category", "?")
    note = (
        f"find_pois for category '{category}' only returned {len(result)} result(s) "
        f"within {current_radius}m. Consider widening the radius to {new_radius}m "
        f"and calling find_pois again for '{category}' before proceeding."
    )
    logger.info("re-planning decision: %s", note)
    return note


def build_agent_graph(tools: list):
    """Compile the LangGraph agent graph, given the list of available tools
    (MCP tools discovered from the backend + the local candidate-grid tool)."""

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def agent_node(state: MessagesState) -> dict:
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        if not response.tool_calls:
            logger.info("final recommendation: %s", response.content[:500])
        return {"messages": [response]}

    async def tools_node(state: MessagesState) -> dict:
        last_message = state["messages"][-1]
        new_messages: list = []
        replan_notes: list[str] = []

        for call in last_message.tool_calls:
            name, args, call_id = call["name"], call["args"], call["id"]
            logger.info("agent tool call: %s(%s)", name, _summarize_args(args))

            tool = tools_by_name.get(name)
            try:
                raw_result = await tool.ainvoke(args)
                parsed_result = _extract_tool_result(raw_result)
            except Exception as exc:
                logger.error("agent tool call failed: %s(%s)", name, args, exc_info=True)
                new_messages.append(ToolMessage(
                    content=f"ERROR calling {name}: {exc}", name=name, tool_call_id=call_id,
                ))
                continue

            logger.info("agent tool result: %s -> %s", name, _summarize_result(name, parsed_result))

            content = parsed_result if isinstance(parsed_result, str) else json.dumps(parsed_result, default=str)
            new_messages.append(ToolMessage(content=content, name=name, tool_call_id=call_id))

            note = _check_replan(name, args, parsed_result)
            if note:
                replan_notes.append(note)

        if replan_notes:
            new_messages.append(HumanMessage(content="\n".join(replan_notes)))

        return {"messages": new_messages}

    def route_after_agent(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def build_initial_messages(brief: str) -> list:
    logger.info("user brief received: %s", brief)
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=brief)]
