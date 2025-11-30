"""
graph.py

Full LangGraph orchestration layer for the Agentic Voice Assistant.

This file defines:
- Unified conversation state for the entire graph
- MCP tool-call wrapper for rag.search and web.search
- Core utilities
- Agents:
    1. Router Agent
    2. Product Discovery Agent (ONLY tool-calling agent)
    3. Comparison Agent
    4. Clarification Agent
    5. Response Synthesizer
- Full LangGraph assembly:
    START → Router → (PD / Comparison / Clarification) → Synthesizer → END
"""

import json
import subprocess
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from schemas import LGState, RagProduct, WebProduct

from config import (
    DEFAULT_MAX_WEB_RESULTS,
    DEFAULT_TOP_K_RAG,
)
from llm.groq_llm import llm_chat

# =====================================================================
# MCP TOOL WRAPPER (PERSISTENT SUBPROCESS)
# =====================================================================

"""
We communicate with the MCP server via a persistent subprocess using stdin/stdout.

The MCP server exposes two tools:

    - rag.search
    - web.search

This wrapper:
- starts mcp_server.py as a background process on first use
- reuses the same process for all tool calls (no re-launch per call)
- sends JSON lines to MCP
- reads JSON lines back
"""

_MCP_PROCESS = None  # type: ignore


def _ensure_mcp_process():
    """Start MCP server as a persistent subprocess if not already running."""
    global _MCP_PROCESS
    if _MCP_PROCESS is None or _MCP_PROCESS.poll() is not None:
        # Launch mcp_server.py once, keep it running
        _MCP_PROCESS = subprocess.Popen(
            ["python", "mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )


def call_mcp_tool(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a tool_call request to the MCP server process.

    Expects mcp_server.py to implement:
        {"cmd": "call", "tool": "...", "payload": {...}}

    Returns:
        Parsed JSON response from MCP, or {"error": "..."} on failure.
    """
    _ensure_mcp_process()
    request = {
        "cmd": "call",
        "tool": tool_name,
        "payload": payload,
    }

    try:
        line = json.dumps(request)
        _MCP_PROCESS.stdin.write(line + "\n")
        _MCP_PROCESS.stdin.flush()

        raw_output = _MCP_PROCESS.stdout.readline().strip()
        if not raw_output:
            return {"error": "Empty response from MCP server."}

        response = json.loads(raw_output)
        return response
    except Exception as e:
        return {"error": f"MCP communication error: {e}"}


# =====================================================================
# STATE INITIALIZATION & HELPERS
# =====================================================================


def initialize_state() -> LGState:
    """
    Create a fresh LangGraph state dictionary.

    Keys:
        user_utterance
        intent
        last_query
        user_preferences
        last_rag_results
        last_web_results
        response_text
    """
    return {
        "user_utterance": None,
        "intent": None,
        "last_query": None,
        "user_preferences": {},
        "last_rag_results": [],
        "last_web_results": [],
        "response_text": None,
    }


def extract_intent(llm_output: str) -> str:
    """
    Basic heuristic to map LLM text to an intent label.
    """
    low = llm_output.lower()

    if "compare" in low or "comparison" in low:
        return "compare"

    if "clarify" in low or "not sure" in low or "preference" in low or "budget" in low:
        return "clarification"

    # Default
    return "product_discovery"


def format_product_card(product: RagProduct) -> str:
    """
    Simple formatter for catalog (private) products.
    """
    out = f"🛒 **{product.title}**\n"
    out += f"- SKU: {product.sku}\n"
    if product.price is not None:
        out += f"- Price: ${product.price}\n"
    if product.brand:
        out += f"- Brand: {product.brand}\n"
    if product.model_number:
        out += f"- Model: {product.model_number}\n"
    if product.shipping_weight_lbs:
        out += f"- Weight: {product.shipping_weight_lbs} lbs\n"
    out += "- Source: Catalog (Private Amazon-2020)\n"
    return out


def format_web_product_card(product: WebProduct) -> str:
    """
    Simple formatter for LIVE web results.
    """
    out = f"🌐 **{product.title}**\n"
    out += "- Source: Live Web (Serper)\n"
    out += f"- URL: {product.url}\n"
    if product.price is not None:
        out += f"- Live Price: ${product.price}\n"
    if product.availability:
        out += f"- Availability: {product.availability}\n"
    return out


# =====================================================================
# PART 2 — ROUTER AGENT (UPDATED, CORRECTED)
# =====================================================================

"""
Router Agent:

- First node after ASR → Final Transcript.
- Reads: state["user_utterance"]
- Classifies intent:
    - product_discovery
    - compare
    - clarification
- Updates:
    - state["intent"]
    - state["last_query"]

CRITICAL RULE FOR COMPARISON:
- Comparison Agent MUST NOT run unless product sets already exist.
"""


def router_agent(state: LGState) -> LGState:
    """
    Router Agent

    Input: state["user_utterance"]
    Output: state with:
        - intent
        - last_query
    """

    user_text = state.get("user_utterance", "") or ""

    # If empty input → default to product discovery
    if not user_text.strip():
        state["intent"] = "product_discovery"
        state["last_query"] = ""
        return state

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Router Agent.\n"
                "Classify the user's intent using only the text.\n"
                "Choose one of:\n"
                "- product_discovery  (user wants help finding products)\n"
                "- compare            (user wants comparison of items already found)\n"
                "- clarification      (user unsure, talking about budget/preferences)\n"
                "Respond in natural language. Do NOT call tools."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    llm_output = llm_chat(messages)
    intent = extract_intent(llm_output)

    state["intent"] = intent
    state["last_query"] = user_text

    return state


def router_decision(state: LGState) -> str:
    """
    Determines the next node in the graph based on state["intent"].

    Return values MUST be node names:
        - "product_discovery_agent"
        - "comparison_agent"
        - "clarification_agent"

    CRITICAL LOGIC:
    - Comparison only allowed if state has previous results.
    """

    intent = state.get("intent", "product_discovery")

    # Comparison: only valid if we already have data to compare
    if intent == "compare":
        rag_ok = bool(state.get("last_rag_results"))
        web_ok = bool(state.get("last_web_results"))
        if rag_ok or web_ok:
            return "comparison_agent"
        else:
            # No results yet → forced product discovery
            return "product_discovery_agent"

    if intent == "clarification":
        return "clarification_agent"

    # Default
    return "product_discovery_agent"


# =====================================================================
# PART 3 — PRODUCT DISCOVERY AGENT (ONLY TOOL-CALLING AGENT)
# =====================================================================

"""
Product Discovery Agent (PDA):

- ONLY agent allowed to call MCP tools:
    - rag.search
    - web.search
- Reads:
    - state["last_query"]
- Calls tools with that query.
- Writes:
    - state["last_rag_results"] (List[RagProduct])
    - state["last_web_results"] (List[WebProduct])
- NO FUSION, NO MERGING.
"""


def product_discovery_agent(state: LGState) -> LGState:
    """
    Product Discovery Agent:
    - Uses last_query as search query
    - Uses LLM to propose tool_calls JSON
    - Robustly falls back to deterministic dual-call if JSON parsing fails
    """

    query = state.get("last_query", "") or ""

    # 1) LLM: ask how to call tools, expecting JSON
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Product Discovery Agent.\n"
                "You MUST call the following TWO tools:\n"
                "- rag.search  (private catalog)\n"
                "- web.search  (live web data)\n\n"
                "Return STRICT JSON ONLY in this format:\n"
                "{\n"
                '  "tool_calls": [\n'
                '    {"tool_name": "rag.search", "payload": {"query": "...", "top_k": 10}},\n'
                '    {"tool_name": "web.search", "payload": {"query": "...", "max_results": 5}}\n'
                "  ]\n"
                "}\n\n"
                "Do NOT write natural language outside JSON.\n"
            ),
        },
        {"role": "user", "content": f"Search for products based on: '{query}'."},
    ]

    llm_out = llm_chat(messages)

    # 2) Parse JSON → tool_calls, else fallback to deterministic calls
    tool_calls: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(llm_out)
        tool_calls = parsed.get("tool_calls", [])
    except Exception:
        tool_calls = []

    if not tool_calls:
        # Fallback: deterministic calls
        tool_calls = [
            {
                "tool_name": "rag.search",
                "payload": {"query": query, "top_k": DEFAULT_TOP_K_RAG},
            },
            {
                "tool_name": "web.search",
                "payload": {"query": query, "max_results": DEFAULT_MAX_WEB_RESULTS},
            },
        ]

    # 3) Execute MCP tools
    rag_results_raw = []
    web_results_raw = []

    for call in tool_calls:
        tool = call.get("tool_name")
        payload = call.get("payload", {})

        if not tool:
            continue

        response = call_mcp_tool(tool, payload)
        if "error" in response:
            # Could log or store error info if needed
            continue

        results = response.get("results", [])

        if tool == "rag.search":
            rag_results_raw = results
        elif tool == "web.search":
            web_results_raw = results

    # 4) Convert raw dicts → RagProduct / WebProduct
    rag_objs: List[RagProduct] = []
    for item in rag_results_raw:
        rag_objs.append(
            RagProduct(
                sku=item["sku"],
                title=item["title"],
                price=item["price"],
                rating=item["rating"],
                brand=item["brand"],
                ingredients=item["ingredients"],
                doc_id=item["doc_id"],
                shipping_weight_lbs=item["shipping_weight_lbs"],
                model_number=item["model_number"],
                raw_metadata=item.get("raw_metadata", {}),
            )
        )

    web_objs: List[WebProduct] = []
    for item in web_results_raw:
        web_objs.append(
            WebProduct(
                title=item["title"],
                url=item["url"],
                snippet=item["snippet"],
                price=item["price"],
                availability=item["availability"],
                source=item["source"],
                raw=item.get("raw", {}),
            )
        )

    # 5) Store in state
    state["last_rag_results"] = rag_objs
    state["last_web_results"] = web_objs

    # Product Discovery Agent does NOT itself set response_text;
    # Response Synthesizer will do that.
    return state


# =====================================================================
# PART 4 — COMPARISON AGENT (NO TOOL CALLS)
# =====================================================================

"""
Comparison Agent:

- Reads:
    - state["last_rag_results"]
    - state["last_web_results"]
- Produces a natural-language comparison summary.
- Does NOT call tools.
- Does NOT merge data.
"""


def comparison_agent(state: LGState) -> LGState:
    rag_items = state.get("last_rag_results", [])
    web_items = state.get("last_web_results", [])

    if not rag_items and not web_items:
        state["response_text"] = (
            "I couldn't find any products to compare yet. "
            "Try asking me to find products first, then we can compare them."
        )
        return state

    lines: List[str] = []
    lines.append("🔍 **Comparison Summary**\n")

    # Catalog side
    if rag_items:
        lines.append("🛒 **Catalog Results (Private Amazon-2020 Dataset):**")
        for item in rag_items[:3]:
            price_str = f"${item.price}" if item.price is not None else "price unknown"
            brand_str = item.brand or "Unknown brand"
            lines.append(f"- {item.title} — {price_str} — {brand_str}")
    else:
        lines.append("🛒 No catalog items available to compare.")

    lines.append("")

    # Web side
    if web_items:
        lines.append("🌐 **Live Web Results (Current Market):**")
        for item in web_items[:3]:
            price_str = f"${item.price}" if item.price is not None else "price unknown"
            avail_str = f" ({item.availability})" if item.availability else ""
            lines.append(f"- {item.title} — {price_str}{avail_str}")
    else:
        lines.append("🌐 No live web items available to compare.")

    lines.append("")
    lines.append(
        "These two sets are separate: catalog items come from your offline private dataset, "
        "while web items reflect up-to-date public information."
    )

    state["response_text"] = "\n".join(lines)
    return state


# =====================================================================
# PART 5 — CLARIFICATION AGENT (NO TOOL CALLS)
# =====================================================================

"""
Clarification Agent:

- Reads:
    - state["user_utterance"]
- Uses LLM to extract preference hints:
    - budget
    - category
    - brand
    - size
    - other
- Updates:
    - state["user_preferences"]
- Writes:
    - state["response_text"] (follow-up question)
- Does NOT call tools.
- After this, graph loops back to Router Agent.
"""


def clarification_agent(state: LGState) -> LGState:
    user_text = state.get("user_utterance", "") or ""
    prefs = state.get("user_preferences", {})

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Clarification Agent.\n"
                "Extract user preferences from the message. Look for:\n"
                "- budget or price limit\n"
                "- product category\n"
                "- brand preference\n"
                "- size/dimensions\n"
                "- anything else meaningful\n\n"
                "Return STRICT JSON ONLY:\n"
                "{\n"
                '  "budget": <number or null>,\n'
                '  "category": <string or null>,\n'
                '  "brand": <string or null>,\n'
                '  "size": <string or null>,\n'
                '  "other": <string or null>\n'
                "}\n"
            ),
        },
        {"role": "user", "content": user_text},
    ]

    llm_out = llm_chat(messages)

    try:
        extracted = json.loads(llm_out)
    except Exception:
        extracted = {
            "budget": None,
            "category": None,
            "brand": None,
            "size": None,
            "other": None,
        }

    for key, value in extracted.items():
        if value is not None and value != "":
            prefs[key] = value

    state["user_preferences"] = prefs

    followup = (
        "Got it, I’ve updated your preferences. "
        "If there’s anything else—like exact size, color, or brand—you care about, tell me, "
        "and I’ll refine the search."
    )
    state["response_text"] = followup

    return state


def clarification_decision(_state: LGState) -> str:
    """
    After Clarification Agent, we always route back to Router Agent.
    """
    return "router_agent"


# =====================================================================
# PART 6 — RESPONSE SYNTHESIZER
# =====================================================================

"""
Response Synthesizer:

- Final agent before TTS/UI.
- Reads:
    - state["last_rag_results"]
    - state["last_web_results"]
    - state["response_text"] (if set by Comparison or Clarification)
- Produces final message for:
    - Chat window
    - TTS (speech)
- NEVER calls tools.
- Keeps catalog vs web clearly separated.
"""


def response_synthesizer(state: LGState) -> LGState:
    rag_items = state.get("last_rag_results", [])
    web_items = state.get("last_web_results", [])
    base_text = state.get("response_text", "")

    lines: List[str] = []

    if base_text:
        lines.append(base_text)
        lines.append("")  # spacing

    # Catalog product cards
    if rag_items:
        lines.append("🛒 **Catalog Items (Private Amazon-2020 Dataset):**")
        for item in rag_items[:3]:
            lines.append(format_product_card(item))
        lines.append("")
    else:
        if not base_text:
            lines.append("🛒 No catalog items found.\n")

    # Web product cards
    if web_items:
        lines.append("🌐 **Live Web Items:**")
        for item in web_items[:3]:
            lines.append(format_web_product_card(item))
        lines.append("")
    else:
        if not base_text:
            lines.append("🌐 No live web items found.\n")

    if not base_text:
        lines.append(
            "These products come from two separate sources: "
            "catalog results are offline private data, and web results are live public data."
        )

    state["response_text"] = "\n".join(lines).strip()
    return state


# =====================================================================
# PART 7 — LANGGRAPH ASSEMBLY
# =====================================================================

"""
Graph layout:

START
  ↓
router_agent
  ↓ (via router_decision)
  ├── product_discovery_agent → response_synthesizer → END
  ├── comparison_agent        → response_synthesizer → END
  └── clarification_agent     → router_agent (loop)
"""

graph = StateGraph(LGState)

graph.add_node("router_agent", router_agent)
graph.add_node("product_discovery_agent", product_discovery_agent)
graph.add_node("comparison_agent", comparison_agent)
graph.add_node("clarification_agent", clarification_agent)
graph.add_node("response_synthesizer", response_synthesizer)

# START → Router
graph.add_edge(START, "router_agent")

# Router → next node (keys MUST match router_decision return values)
graph.add_conditional_edges(
    "router_agent",
    router_decision,
    {
        "product_discovery_agent": "product_discovery_agent",
        "comparison_agent": "comparison_agent",
        "clarification_agent": "clarification_agent",
    },
)

# Product Discovery → Synthesizer → END
graph.add_edge("product_discovery_agent", "response_synthesizer")
graph.add_edge("comparison_agent", "response_synthesizer")
graph.add_edge("response_synthesizer", END)

# Clarification → Router loop
graph.add_conditional_edges(
    "clarification_agent",
    clarification_decision,
    {
        "router_agent": "router_agent",
    },
)

workflow = graph.compile()


def run_graph(user_text: str, state: LGState = None) -> LGState:
    """
    Execute the full LangGraph workflow for a single user input.

    - Injects user_text into state["user_utterance"]
    - Runs workflow
    - Returns updated state
    """
    if state is None:
        state = initialize_state()

    state["user_utterance"] = user_text
    new_state = workflow.invoke(state)
    return new_state


if __name__ == "__main__":
    # Quick manual test
    s = initialize_state()
    print("User: What kind of window film should I buy for privacy under $20?\n")
    out = run_graph("What kind of window film should I buy for privacy under $20?", s)
    print("\n=== FINAL OUTPUT ===\n")
    print(out["response_text"])
