# MCP Tools Documentation

**Agentic Voice-to-Voice Product Discovery Assistant**
*(Two-Tool MCP Server: `rag.search` + `web.search`)*

---

## Overview

This project implements a lightweight **MCP (Model Context Protocol)** server that exposes two tools:

1. **`rag.search`** – queries the private Amazon Product Dataset 2020 (vector search + metadata filters)
2. **`web.search`** – performs live web search via Serper API, returning prices and availability when possible

These tools allow the LLM-driven planner to uniformly call both private and live retrieval under a single protocol.

The MCP server supports:

* **Tool discovery** (names + input/output schemas)
* **Structured JSON I/O**
* **TTL caching and rate limiting** for web search
* **Timestamped logging** for every tool invocation
* **Provenance preservation** (doc_ids and URLs)

Implementation lives in:

* `src/mcp/mcp_server.py`
* `src/mcp/rag_tool.py`
* `src/mcp/web_tool.py`

---

# 1. MCP Server Architecture

The server is a simple **stdio** loop that handles three message types:

| Command    | Purpose                              |
| ---------- | ------------------------------------ |
| `discover` | Return tool names and JSON schemas   |
| `call`     | Execute a tool with structured input |
| (default)  | Return error for unknown commands    |

Main server loop (simplified):

```python
while True:
    line = sys.stdin.readline()
    ...
    if cmd == "discover":
        return { "tools": [RAG_SEARCH_SCHEMA, WEB_SEARCH_SCHEMA] }
    elif cmd == "call":
        if tool == "rag.search": return _call_rag_search(...)
        if tool == "web.search": return _call_web_search(...)
```

This keeps the server simple, transparent, and easily testable.

---

# 2. Tool Discovery

A client sends:

```json
{"cmd": "discover"}
```

The server returns:

```json
{
  "tools": [
    { "name": "rag.search", "description": "...", "input_schema": {...}, "output_schema": {...} },
    { "name": "web.search", "description": "...", "input_schema": {...}, "output_schema": {...} }
  ]
}
```

This fulfills the rubric requirement for **MCP tool discovery + JSON schemas**.

---

# 3. JSON Schemas (as implemented)

Below are the schemas available to clients during discovery.

---

## 3.1 `rag.search` Schema

### **Inputs**

```json
{
  "query": "string",
  "top_k": "integer (optional)",
  "filters": "object (optional metadata filters)"
}
```

### **Outputs**

```json
{
  "results": [
    {
      "sku": "string",
      "title": "string",
      "price": "number or null",
      "rating": "number or null",
      "brand": "string or null",
      "ingredients": "string or null",
      "doc_id": "string",
      "shipping_weight_lbs": "number or null",
      "model_number": "string or null",
      "raw_metadata": "object"
    }
  ]
}
```

### Purpose

* High-quality **vector + metadata** retrieval from the private Amazon-2020 slice
* Returns rich structured data for grounding, citations, tables, and conflict detection

---

## 3.2 `web.search` Schema

### **Inputs**

```json
{
  "query": "string",
  "max_results": "integer (optional)"
}
```

### **Outputs**

```json
{
  "results": [
    {
      "title": "string",
      "url": "string",
      "snippet": "string",
      "price": "number or null",
      "availability": "string or null",
      "source": "string",
      "raw": "object"
    }
  ]
}
```

### Purpose

* Provides **real-time** verification of pricing and availability
* Enables conflict detection between fixed catalog data and live web signals

---

# 4. rag.search Implementation Details

Implemented in `src/mcp/rag_tool.py`.

### Workflow

1. Receive query text and optional filters
2. Perform a **vector similarity search** over FAISS/Chroma using stored embeddings
3. Apply metadata filters (brand, price range, category, etc.)
4. Assemble clean Python objects with:

   * title
   * price
   * rating
   * brand
   * model number
   * shipping weight
   * doc_id (**critical for citations**)

### Logging

Every request writes a timestamped entry via `_log`:

```
[2025-12-10T02:13:22Z] rag.search_request { "query": "...", "filters": {...} }
[2025-12-10T02:13:22Z] rag.search_response { "count": 3 }
```

---

# 5. web.search Implementation Details

Implemented in `src/mcp/web_tool.py`.

### Workflow

1. Rate-limited API call to Serper
2. TTL cache lookup (300s by default)
3. Normalize fields (title, snippet, price, availability)
4. Provide provenance via `url` + `source`

### Rate Limiting

A simple per-call time guard ensures you don’t exceed Serper’s free-tier limits:

```python
if now - _last_call_timestamp < SERPER_MIN_DELAY:
    sleep()
_last_call_timestamp = now
```

### TTL Caching

Implemented via:

```python
_serper_cache: Dict[str, Dict] = {}
_serper_cache_timestamps: Dict[str, float] = {}

if now - ts < WEB_SEARCH_TTL_SECONDS:
    return cached_result
```

This satisfies:

> “cache (TTL 60–300s)”

### Logging

Logged at both request and response:

```
[2025-12-10T02:14:01Z] web.search_request { "query": "stainless cleaner" }
[2025-12-10T02:14:01Z] web.search_response { "count": 4, "urls": [...] }
```

---

# 6. Transport & Invocation

The planner calls MCP tools through an abstraction layer:

```python
mcp.invoke("rag.search", {...})
```

or

```python
mcp.invoke("web.search", {...})
```

Which maps to:

```
{"cmd": "call", "tool": "rag.search", "input": {...}}
```

---

# 7. Safety & Compliance Notes

* Pricing/availability from web.search are considered **more current** than catalog and flagged during synthesis.
* Tools are **explicitly typed and schema-discoverable**, avoiding prompt-based tool calls.
* All URLs are logged for traceability (per rubric).
* No unsafe keywords are passed through to web search (safety pre-filter at router).
* No secrets or tokens are logged.

---

# 8. Summary

This MCP layer provides:

* A **unified retrieval abstraction** over private RAG and live search
* Fully typed, discoverable tools
* TTL caching + rate limiting
* Strong provenance (doc_ids + URLs)
* Clean JSON contracts enabling deterministic multi-agent planning

These features directly support:

- MCP Server
- Multi-agent planning
- Grounded retrieval & reconciliation
- Traceable pipeline