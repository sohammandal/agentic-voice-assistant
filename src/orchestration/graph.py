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
import re
import subprocess
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from src.config import (
    DEFAULT_MAX_WEB_RESULTS,
    DEFAULT_TOP_K_RAG,
    PROJECT_ROOT,
)
from src.llm.groq_llm import llm_chat

from ..mcp.schemas import LGState, RagProduct, WebProduct

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
        _MCP_PROCESS = subprocess.Popen(
            ["python", "-m", "src.mcp.mcp_server"],
            cwd=str(PROJECT_ROOT),
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
        step_log
        plan
        safety_flags
        price_conflicts
    """
    return {
        "user_utterance": None,
        "intent": None,
        "last_query": None,
        "user_preferences": {},
        "last_rag_results": [],
        "last_web_results": [],
        "response_text": None,
        "step_log": [],
        "plan": {},
        "safety_flags": {},
        "price_conflicts": [],  # catalog vs web price differences
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


def _infer_brand_from_title(title: str) -> str:
    """
    Fallback: infer brand as the leading token(s) of the title until
    we hit a token that contains a digit.

    Example:
        'd-c-fix 96445 Spring Chapel/Tulia Window Film ...'
        -> 'd-c-fix'
    """
    if not title:
        return ""
    tokens = title.split()
    brand_tokens: List[str] = []
    for tok in tokens:
        if any(ch.isdigit() for ch in tok):
            break
        brand_tokens.append(tok)
    return " ".join(brand_tokens).strip()


def format_product_card(product: RagProduct) -> str:
    """
    Simple formatter for catalog (private) products.
    """
    out = f"🛒 **{product.title}**\n"
    out += f"- SKU: {product.sku}\n"
    if product.price is not None:
        out += f"- Price: ${product.price}\n"

    # Use stored brand, but if it's missing or suspiciously short,
    # fall back to inferring from the title.
    brand = (product.brand or "").strip()
    if not brand or len(brand) <= 1:
        inferred = _infer_brand_from_title(product.title or "")
        if inferred:
            brand = inferred

    if brand:
        out += f"- Brand: {brand}\n"

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


def detect_price_conflicts(catalog_items: list, web_items: list) -> list[dict]:
    """
    Very simple conflict detector:
    - tries to match items by overlapping brand or title text
    - flags when price differs more than 10 percent and 0.50 units
    Returns a list of conflict dicts.
    """
    conflicts: list[dict] = []

    if not catalog_items or not web_items:
        return conflicts

    def norm(s: str) -> str:
        return (s or "").lower().strip()

    for cat in catalog_items:
        c_title = norm(cat.get("title", ""))
        c_brand = norm(cat.get("brand", ""))
        c_price = cat.get("price")
        if c_price is None:
            continue

        for web in web_items:
            w_title = norm(web.get("title", ""))
            w_brand = norm(web.get("brand", ""))
            w_price = web.get("price")
            if w_price is None:
                continue

            # Very cheap matching rule:
            title_overlap = (
                c_title and c_title[:20] in w_title or w_title[:20] in c_title
            )
            brand_overlap = c_brand and (c_brand in w_title or c_brand == w_brand)

            if not (title_overlap or brand_overlap):
                continue

            diff = abs(float(w_price) - float(c_price))
            rel_diff = diff / max(float(c_price), 1.0)

            if diff >= 0.5 and rel_diff >= 0.10:
                conflicts.append(
                    {
                        "sku": cat.get("sku"),
                        "title": cat.get("title"),
                        "catalog_price": float(c_price),
                        "web_price": float(w_price),
                        "web_url": web.get("url"),
                    }
                )
                break  # stop after first web match for this catalog item

        if len(conflicts) >= 3:
            break  # cap to a few so the prompt does not explode

    return conflicts


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
    user_text = state.get("user_utterance", "") or ""
    lower = user_text.lower()

    # Initialize or reuse safety flags
    safety = state.get("safety_flags", {}) or {}

    # Very simple allowlist / blocklist for dangerous product domains
    banned_keywords = [
        "explosive",
        "bomb",
        "pesticide",
        "poison",
        "rat poison",
        "cyanide",
        "molotov",
        "weapon",
        "gunpowder",
    ]

    if any(k in lower for k in banned_keywords):
        safety["blocked"] = True
        safety["reason"] = "disallowed or dangerous product category"
    else:
        safety["blocked"] = False
        safety["reason"] = None

    state["safety_flags"] = safety

    # If blocked, short circuit with a safe response
    if safety.get("blocked"):
        state["intent"] = "blocked"
        state["last_query"] = user_text
        state["response_text"] = (
            "I can’t help with that request. Please ask about safe, legal household items instead."
        )
        log = state.get("step_log", [])
        log.append("Router blocked unsafe request based on keyword allowlist.")
        state["step_log"] = log
        return state

    # If empty input -> default to product discovery
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

    # Heuristic override based on raw user text
    u = user_text.lower()

    # If user clearly wants recommendations, force product discovery
    if any(
        p in u for p in ["recommend", "suggest", "find me", "show me", "help me find"]
    ):
        intent = "product_discovery"

    # If user clearly wants comparison, force compare
    if any(p in u for p in ["compare", " vs ", " versus "]):
        intent = "compare"

    # Extract numeric budget like "under 15 dollars", "under $15"
    prefs = state.get("user_preferences", {}) or {}
    m = re.search(r"under\s+\$?(\d+(?:\.\d+)?)", u)
    if m:
        try:
            prefs["budget"] = float(m.group(1))
        except ValueError:
            pass
    state["user_preferences"] = prefs

    state["intent"] = intent
    state["last_query"] = user_text

    log = state.get("step_log", [])
    log.append(f"Router decided intent = {intent}")
    state["step_log"] = log

    return state


def router_decision(state: LGState) -> str:
    """
    Determines the next node in the graph based on state["intent"].

    Return values MUST be node names:
        - "planner_agent"
        - "comparison_agent"
        - "clarification_agent"

    CRITICAL LOGIC:
    - Comparison only allowed if state has previous results.
    """

    intent = state.get("intent", "product_discovery")

    # If the router flagged this as unsafe, jump straight to the final response
    safety = state.get("safety_flags", {}) or {}
    if safety.get("blocked"):
        return "response_synthesizer"

    # Comparison: only valid if we already have data to compare
    if intent == "compare":
        rag_ok = bool(state.get("last_rag_results"))
        web_ok = bool(state.get("last_web_results"))
        if rag_ok or web_ok:
            return "comparison_agent"
        else:
            # No results yet -> plan and then discover
            return "planner_agent"

    if intent == "clarification":
        return "clarification_agent"

    # Default: plan first
    return "planner_agent"


def planner_agent(state: LGState) -> LGState:
    """
    Planner Agent.

    Decides:
      - whether to use rag.search and/or web.search
      - basic metadata filters for RAG (e.g. category, budget)

    Writes its plan into state["plan"] as a dict like:
      {
        "use_rag": true,
        "use_web": true,
        "rag_top_k": 10,
        "web_max_results": 5,
        "rag_filters": {"category": "curtain"},
        "reasoning": "User wants eco-friendly cleaner under $15, so search both catalog and live web"
      }
    """

    user_text = state.get("user_utterance", "") or ""
    prefs = state.get("user_preferences", {}) or {}

    system_prompt = (
        "You are the Planner agent in a shopping assistant.\n"
        "Given the user's request and known preferences, decide which sources to use:\n"
        "- Private catalog via rag.search\n"
        "- Live web via web.search\n"
        "Also provide reasoning for your decisions.\n\n"
        "Return STRICT JSON ONLY in this format:\n"
        "{\n"
        '  "use_rag": true,\n'
        '  "use_web": true,\n'
        '  "rag_top_k": 10,\n'
        '  "web_max_results": 5,\n'
        '  "reasoning": "User wants eco-friendly cleaner under $15, so search both catalog and live web for best options"\n'
        "}\n"
        "If you are unsure about filters, set rag_filters to an empty object {}.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User request: {user_text}\nKnown preferences: {json.dumps(prefs)}"
            ),
        },
    ]

    plan: Dict[str, Any] = {
        "use_rag": True,
        "use_web": True,
        "rag_top_k": DEFAULT_TOP_K_RAG,
        "web_max_results": DEFAULT_MAX_WEB_RESULTS,
        "rag_filters": {},
        "reasoning": "Default plan: search both catalog and live web",
    }

    try:
        llm_out = llm_chat(messages, max_tokens=256)
        parsed = json.loads(llm_out)
        if isinstance(parsed, dict):
            plan.update(parsed)
    except Exception:
        # keep default plan
        pass

    # --- normalize rag_filters ---
    prefs = state.get("user_preferences", {}) or {}
    budget = prefs.get("budget")

    clauses = []

    # 1) Constrain to the catalog slice we care about
    clauses.append({"main_category": {"$eq": "home & kitchen"}})

    # 2) Apply budget if we have one
    if isinstance(budget, (int, float)):
        clauses.append({"price": {"$lte": float(budget)}})

    # 3) Emit Chroma-compatible filters
    if clauses:
        plan["rag_filters"] = {"$and": clauses}
    else:
        plan["rag_filters"] = {}
    # --- end normalize ---

    state["plan"] = plan

    log = state.get("step_log", [])
    log.append(
        f"Planner decided use_rag={plan.get('use_rag')} "
        f"use_web={plan.get('use_web')} "
        f"rag_filters={plan.get('rag_filters')} | "
        f"Reasoning: {plan.get('reasoning', 'N/A')}"
    )
    state["step_log"] = log

    return state


def planner_decision(_state: LGState) -> str:
    """
    Planner always routes to Product Discovery for now.
    Later you could add branches (e.g. web_only).
    """
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

    # 2) Read Planner plan
    plan = state.get("plan", {}) or {}
    use_rag = plan.get("use_rag", True)
    use_web = plan.get("use_web", True)
    rag_top_k = plan.get("rag_top_k", DEFAULT_TOP_K_RAG)
    web_max_results = plan.get("web_max_results", DEFAULT_MAX_WEB_RESULTS)
    rag_filters = plan.get("rag_filters", {}) or {}

    # 3) Parse JSON -> tool_calls
    tool_calls: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(llm_out)
        tool_calls = parsed.get("tool_calls", [])
    except Exception:
        tool_calls = []

    # 3a) If LLM gave us something, normalize it to respect the plan
    normalized: List[Dict[str, Any]] = []
    for call in tool_calls:
        name = call.get("tool_name")
        payload = call.get("payload", {}) or {}

        if name == "rag.search":
            if not use_rag:
                continue
            payload.setdefault("query", query)
            payload["top_k"] = rag_top_k
            if rag_filters:
                payload["filters"] = rag_filters
            normalized.append({"tool_name": "rag.search", "payload": payload})

        elif name == "web.search":
            if not use_web:
                continue
            payload.setdefault("query", query)
            payload["max_results"] = web_max_results
            normalized.append({"tool_name": "web.search", "payload": payload})

    tool_calls = normalized

    # 3b) If LLM JSON failed or got filtered out, fall back to deterministic plan-based calls
    if not tool_calls:
        tool_calls = []
        if use_rag:
            payload = {"query": query, "top_k": rag_top_k}
            if rag_filters:
                payload["filters"] = rag_filters
            tool_calls.append({"tool_name": "rag.search", "payload": payload})
        if use_web:
            tool_calls.append(
                {
                    "tool_name": "web.search",
                    "payload": {"query": query, "max_results": web_max_results},
                }
            )

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

    # 6) Detect price conflicts between catalog and web results
    conflicts_input_catalog = [
        {
            "sku": p.sku,
            "title": p.title,
            "brand": p.brand,
            "price": p.price,
        }
        for p in rag_objs
        if p.price is not None
    ]

    conflicts_input_web = [
        {
            "title": w.title,
            # try to pull brand from raw if available; ok if this is None
            "brand": (w.raw or {}).get("brand") if isinstance(w.raw, dict) else None,
            "price": w.price,
            "url": w.url,
        }
        for w in web_objs
        if w.price is not None
    ]

    conflicts = detect_price_conflicts(conflicts_input_catalog, conflicts_input_web)
    state["price_conflicts"] = conflicts

    log = state.get("step_log", [])
    if conflicts:
        log.append(
            f"ConflictDetector found {len(conflicts)} price conflicts between catalog and web results."
        )
    else:
        log.append("ConflictDetector found no price conflicts.")

    log.append(
        f"ProductDiscovery called rag.search (got {len(rag_objs)} items) and web.search (got {len(web_objs)} items)"
    )
    state["step_log"] = log

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

    log = state.get("step_log", [])
    log.append(
        f"ComparisonAgent compared {len(rag_items)} catalog items and {len(web_items)} web items"
    )
    state["step_log"] = log

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

    log = state.get("step_log", [])
    log.append("ClarificationAgent updated user_preferences")
    state["step_log"] = log

    return state


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

    # Simple relevance filter for catalog items based on query text
    query_text = (state.get("last_query") or "").lower()

    if query_text and rag_items:
        filtered_rag: List[RagProduct] = []
        # Focus on meaningful tokens (length > 3)
        tokens = [t for t in query_text.split() if len(t) > 3]

        for p in rag_items:
            # Combine title, brand, and a bit of metadata into a single text blob
            meta = getattr(p, "raw_metadata", {}) or {}
            haystack = f"{p.title} {p.brand} {meta.get('features', '')}".lower()

            # Count how many query tokens appear in this product
            matches = sum(1 for tok in tokens if tok in haystack)

            # Keep products that match at least 2 tokens, or 1 if query is very short
            if (len(tokens) >= 2 and matches >= 2) or (
                len(tokens) == 1 and matches >= 1
            ):
                filtered_rag.append(p)

        # Only replace if we actually found something
        if filtered_rag:
            rag_items = filtered_rag

    # Clarification turns: just return the clarification text
    if state.get("intent") == "clarification":
        if not base_text:
            base_text = "I’ve updated your preferences."
        state["response_text"] = base_text
        return state

    # Budget-aware filtering for web results
    prefs = state.get("user_preferences", {}) or {}
    budget = prefs.get("budget")

    if budget is not None:
        filtered_web: List[WebProduct] = []
        for item in web_items:
            # Strict: skip unknown prices when user gave a budget
            if item.price is None:
                continue
            if item.price <= budget:
                filtered_web.append(item)
        web_items = filtered_web

    # Build compact JSON summary for the Answerer LLM
    summary_payload = {
        "original_query": state.get("last_query"),
        "budget": budget,
        "catalog_items": [],
        "web_items": [],
    }

    # Populate catalog items with richer metadata
    for p in rag_items[:3]:
        meta = getattr(p, "raw_metadata", {}) or {}

        brand = (p.brand or "").strip()
        if not brand or len(brand) <= 1:
            inferred = _infer_brand_from_title(p.title or "")
            if inferred:
                brand = inferred

        summary_payload["catalog_items"].append(
            {
                "title": p.title,
                "price": p.price,
                "brand": brand,
                "rating": p.rating,
                # NEW: richer info (unchanged below)
                "model_number": p.model_number,
                "shipping_weight_lbs": p.shipping_weight_lbs,
                "features": meta.get("features"),
                "category_path": meta.get("category_path"),
                "main_category": meta.get("main_category"),
            }
        )

    # Populate web items the same as before
    for w in web_items[:3]:
        summary_payload["web_items"].append(
            {
                "title": w.title,
                "price": w.price,
                "availability": w.availability,
                "source": w.source,
            }
        )

    # Call Groq to write a short spoken recommendation
    answer_text = ""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Answerer agent for a shopping assistant.\n"
                    "You receive structured data about products from a private catalog "
                    "and from live web search.\n"
                    "Write a concise spoken answer (2 to 4 sentences) recommending up to 2 options.\n"
                    "If both sources have relevant items, briefly compare them on "
                    "factors such as price and features.\n"
                    "Respect the user's budget if provided.\n"
                    "Do not invent details that are not in the data. "
                    "Do not fabricate prices or ratings.\n"
                ),
            },
            {
                "role": "user",
                "content": f"Here is the data: {json.dumps(summary_payload)}",
            },
        ]
        answer_text = llm_chat(messages, max_tokens=256)
    except Exception:
        # Fall back to any existing base_text if LLM fails
        answer_text = base_text or ""

    lines: List[str] = []

    if answer_text:
        lines.append(answer_text)
        lines.append("")

    # Catalog product cards
    if rag_items:
        lines.append("🛒 **Catalog Items (Private Amazon-2020 Dataset):**")
        for item in rag_items[:3]:
            lines.append(format_product_card(item))
        lines.append("")
    else:
        lines.append("🛒 No catalog items found.\n")

    # Web product cards
    if web_items:
        lines.append("🌐 **Live Web Items:**")
        for item in web_items[:3]:
            lines.append(format_web_product_card(item))
        lines.append("")
    else:
        lines.append("🌐 No live web items found.\n")

    # Price conflict summary between catalog and live web
    conflicts = state.get("price_conflicts") or []
    if conflicts:
        lines.append("⚠️ **Price differences between catalog and live web:**")
        for c in conflicts[:3]:
            try:
                catalog_price = float(c.get("catalog_price", 0.0))
                web_price = float(c.get("web_price", 0.0))
                lines.append(
                    f"- {c.get('title', 'Unknown item')}: "
                    f"catalog ${catalog_price:.2f} vs web ${web_price:.2f} "
                    f"(link: {c.get('web_url', 'N/A')})"
                )
            except Exception:
                # If anything is weird, skip formatting that entry
                continue
        lines.append(
            "Prices can legitimately differ because the private catalog is static "
            "while web prices change over time."
        )
        lines.append("")

    # Extra explanation when absolutely nothing was found
    if not rag_items and not web_items:
        lines.append(
            "These products come from two separate sources: "
            "catalog results are offline private data, and web results are live public data."
        )

    state["response_text"] = "\n".join(lines).strip()

    log = state.get("step_log", [])
    conflicts = state.get("price_conflicts") or []
    log.append(
        f"ResponseSynthesizer produced answer using {len(rag_items)} catalog items, "
        f"{len(web_items)} web items, and {len(conflicts)} price conflicts."
    )

    state["step_log"] = log

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
  └── comparison_agent        → response_synthesizer → END
  └── clarification_agent -> response_synthesizer -> END
"""

graph = StateGraph(LGState)

graph.add_node("router_agent", router_agent)
graph.add_node("planner_agent", planner_agent)
graph.add_node("product_discovery_agent", product_discovery_agent)
graph.add_node("comparison_agent", comparison_agent)
graph.add_node("clarification_agent", clarification_agent)
graph.add_node("response_synthesizer", response_synthesizer)

# START → Router
graph.add_edge(START, "router_agent")

# Router -> next node
graph.add_conditional_edges(
    "router_agent",
    router_decision,
    {
        "planner_agent": "planner_agent",
        "comparison_agent": "comparison_agent",
        "clarification_agent": "clarification_agent",
        "response_synthesizer": "response_synthesizer",
    },
)

# Planner -> Product Discovery
graph.add_conditional_edges(
    "planner_agent",
    planner_decision,
    {
        "product_discovery_agent": "product_discovery_agent",
    },
)

# Product Discovery → Synthesizer → END
graph.add_edge("product_discovery_agent", "response_synthesizer")
graph.add_edge("comparison_agent", "response_synthesizer")
graph.add_edge("response_synthesizer", END)

# Clarification → Synthesizer → END
graph.add_edge("clarification_agent", "response_synthesizer")

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
