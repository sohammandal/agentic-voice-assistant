# MCP Server Layer (two tools)

# Purpose: these are the exact two MCP tools:
# rag.search → mcp_rag_search_tool
# web.search → mcp_web_search_tool
# Your actual MCP framework will register these as tools with their input/output JSON schemas.

# Your Two-Tool MCP Server, exactly as required by the project spec:
# ✔ Exposes rag.search
# ✔ Exposes web.search

# ✔ Implements:
# Tool discovery
# JSON schema definitions for both tools
# Transport via standard I/O (stdio)
# Logs requests + responses
# Unified interface for LangGraph agents
# This file is the bridge between LangGraph and both retrieval systems.

# mcp_server.py =
# A small server that listens on stdin, receives “rag.search” or “web.search” commands, executes them, and returns structured JSON results back to your agents.

# It acts as the middleman between:
# LangGraph agents
# Your RAG DB
# Serper web search

# mcp_server.py
"""
mcp_server.py

Implements the required MCP server exposing two retrieval tools:
  - rag.search  (private vector DB over Amazon 2020)
  - web.search  (Serper live web search)

Follows MCP tool protocol:
  - Tool discovery (list names + schemas)
  - Tool call (execute tool and return structured JSON)
  - Stdio transport
  - Logging requests/responses
"""

import datetime
import json
import sys
import traceback
from typing import Any, Dict

# Load environment variables from .env if python-dotenv is available
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv() or ".env")
except Exception:
    # If python-dotenv is not installed, assume env vars are set in the shell
    pass


from .rag_tool import rag_search
from .schemas import RagProduct, WebProduct
from .web_tool import web_search

# ------------------------------------------------------------------------------
# JSON Schemas (used for tool discovery)
# ------------------------------------------------------------------------------
RAG_SEARCH_SCHEMA = {
    "name": "rag.search",
    "description": "Query private Amazon-2020 vector DB for semantically relevant products.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
    },
    "output_schema": {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {"type": "object"}}},
    },
}

WEB_SEARCH_SCHEMA = {
    "name": "web.search",
    "description": "Perform live Serper web search for real-time prices + availability.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
    "output_schema": {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {"type": "object"}}},
    },
}


# ------------------------------------------------------------------------------
# Logging Utility
# ------------------------------------------------------------------------------
def _log(event: str, payload: Dict[str, Any]):
    ts = datetime.datetime.utcnow().isoformat()
    sys.stderr.write(f"[{ts}] {event}: {json.dumps(payload)}\n")
    sys.stderr.flush()


# ------------------------------------------------------------------------------
# Tool Invocation
# ------------------------------------------------------------------------------
def _call_rag_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload["query"]
    top_k = payload.get("top_k", 10)

    _log("rag.search_request", payload)
    results = rag_search(query=query, top_k=top_k)

    out = []
    for r in results:
        out.append(
            {
                "sku": r.sku,
                "title": r.title,
                "price": r.price,
                "rating": r.rating,
                "brand": r.brand,
                "ingredients": r.ingredients,
                "doc_id": r.doc_id,
                "shipping_weight_lbs": r.shipping_weight_lbs,
                "model_number": r.model_number,
                "raw_metadata": r.raw_metadata,
            }
        )

    _log("rag.search_response", {"count": len(out)})
    return {"results": out}


def _call_web_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = payload["query"]
    max_results = payload.get("max_results", 5)

    _log("web.search_request", payload)

    results = web_search(query=query, max_results=max_results)

    out = []
    for w in results:
        out.append(
            {
                "title": w.title,
                "url": w.url,
                "snippet": w.snippet,
                "price": w.price,
                "availability": w.availability,
                "source": w.source,
                "raw": w.raw,
            }
        )

    _log("web.search_response", {"count": len(out)})
    return {"results": out}


# ------------------------------------------------------------------------------
# Tool Router
# ------------------------------------------------------------------------------
def call_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatches a tool call."""
    tool = request["tool"]
    payload = request.get("payload", {})

    try:
        if tool == "rag.search":
            return _call_rag_search(payload)
        elif tool == "web.search":
            return _call_web_search(payload)
        else:
            return {"error": f"Unknown tool: {tool}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ------------------------------------------------------------------------------
# MCP Server Loop (stdio)
# ------------------------------------------------------------------------------
def main():
    """
    MCP main loop:
      - Reads JSON lines from STDIN
      - Writes JSON responses to STDOUT
      - Supports two commands:
            {"cmd": "discover"}
            {"cmd": "call", "tool": "...", "payload": {...}}
    """

    # Startup message to stderr
    sys.stderr.write("[MCP] Server running, listening for commands...\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except:
            continue

        cmd = msg.get("cmd")

        # Tool discovery
        if cmd == "discover":
            response = {
                "tools": [
                    RAG_SEARCH_SCHEMA,
                    WEB_SEARCH_SCHEMA,
                ]
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        # Tool call
        if cmd == "call":
            response = call_tool(msg)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        # Unknown
        sys.stdout.write(json.dumps({"error": "unknown command"}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
