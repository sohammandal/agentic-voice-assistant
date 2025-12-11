# Prompts

This document collects all the main prompts used by the system, grouped by agent and tool.

It is meant to satisfy the Prompt Disclosure requirement in the grading rubric:

* System prompts
* Router and planner tool prompts
* JSON format instructions
* Safety related instructions
* How prompts map to code and agents

The model is configured in:

* `src/config.py`

  * `GROQ_MODEL_NAME = "llama-3.3-70b-versatile"`
  * `LLM_TEMPERATURE` and `LLM_MAX_TOKENS`
* `src/llm/groq_llm.py`

  * `llm_chat` wraps the Groq client

Each section below shows the system prompt and how it is used.

---

## 1. Router Agent

**Code location**

* File: `src/orchestration/graph.py`
* Function: `router_agent`

**Role**

* Classifies the user intent into:

  * `product_discovery`
  * `compare`
  * `clarification`
* Seeds a numeric budget into `user_preferences` via regex
* Applies a simple keyword based safety check for dangerous product domains

**System prompt**

```text
You are the Router Agent.
Classify the user's intent using only the text.
Choose one of:
- product_discovery  (user wants help finding products)
- compare            (user wants comparison of items already found)
- clarification      (user unsure, talking about budget/preferences)
Respond in natural language. Do NOT call tools.
```

**User message format**

```text
{user_utterance}
```

Heuristics in code then override this classification for phrases like:

* "recommend", "suggest", "find me", "show me"  -> force `product_discovery`
* "compare", " vs ", " versus "                -> force `compare`

The safety check is not LLM based. The router uses a small banned keyword list (for example "bomb", "explosive", "weapon") and, if matched, sets:

* `state["safety_flags"]["blocked"] = True`
* `state["response_text"]` to a fixed safety message
* `state["intent"] = "blocked"`

In that case the graph goes straight to the Response Synthesizer without calling tools.

---

## 2. Planner Agent

**Code location**

* File: `src/orchestration/graph.py`
* Function: `planner_agent`

**Role**

* Decide:

  * Whether to use `rag.search` and or `web.search`
  * How many results to fetch from each
  * A high level reasoning string
* Code then normalizes and augments the plan with:

  * Budget based `rag_filters`
  * Category filters (home and kitchen etc)

**System prompt**

```text
You are the Planner agent in a shopping assistant.
Given the user's request and known preferences, decide which sources to use:
- Private catalog via rag.search
- Live web via web.search
Also provide reasoning for your decisions.

Return STRICT JSON ONLY in this format:
{
  "use_rag": true,
  "use_web": true,
  "rag_top_k": 10,
  "web_max_results": 5,
  "reasoning": "User wants eco-friendly cleaner under $15, so search both catalog and live web for best options"
}

If you are unsure about filters, set rag_filters to an empty object {}.
```

**User message format**

```text
User request: {user_utterance}
Known preferences: {user_preferences as JSON}
```

The JSON returned by the model is parsed and merged into a default plan, then further adjusted by code to add actual Chroma filters from the budget and category hints.

---

## 3. Product Discovery Agent (tool calling)

**Code location**

* File: `src/orchestration/graph.py`
* Function: `product_discovery_agent`

**Role**

* This is the only agent allowed to call MCP tools.
* It uses the LLM to suggest tool calls, then the code normalizes them and enforces the planned settings (top_k, filters, etc).
* It calls:

  * `rag.search` for private catalog
  * `web.search` for live web pricing and availability
* After retrieval, code runs a simple conflict detector to compare catalog vs web prices and stores any discrepancies in `state["price_conflicts"]` for downstream use.

**System prompt**

```text
You are the Product Discovery Agent.
You MUST call the following TWO tools:
- rag.search  (private catalog)
- web.search  (live web data)

Return STRICT JSON ONLY in this format:
{
  "tool_calls": [
    {"tool_name": "rag.search", "payload": {"query": "...", "top_k": 10}},
    {"tool_name": "web.search", "payload": {"query": "...", "max_results": 5}}
  ]
}

Do not write natural language outside JSON.
```

**User message format**

```text
Search for products based on: '{last_query}'.
```

The model output is parsed as JSON. The code then:

* Aligns `top_k`, `max_results`, and `filters` with `state["plan"]`
* Ensures at most one call to each tool per turn
* Falls back to a deterministic, plan based tool call if JSON parsing fails

Conflict handling is implemented in code, not in this prompt. The agent:

* Builds light weight views of catalog and web items (sku, title, brand, price, url)
* Matches items by overlapping brand or title text
* Flags a conflict when the web price differs from the catalog price by:

  * at least 0.50 currency units, and
  * at least 10 percent relative difference
* Stores a small list of conflicts in `state["price_conflicts"]` plus a log entry

---

## 4. Clarification Agent

**Code location**

* File: `src/orchestration/graph.py`
* Function: `clarification_agent`

**Role**

* Extracts user preferences when the intent is `clarification`
* Does not call any tools
* Safety is handled upstream in the Router, so this agent only runs for allowed queries

**System prompt**

```text
You are the Clarification Agent.
Extract user preferences from the message. Look for:
- budget or price limit
- product category
- brand preference
- size/dimensions
- anything else meaningful

Return STRICT JSON ONLY:
{
  "budget": <number or null>,
  "category": <string or null>,
  "brand": <string or null>,
  "size": <string or null>,
  "other": <string or null>
}
```

**User message format**

```text
{user_utterance}
```

If JSON parsing fails, the code falls back to a default structure with nulls and leaves existing preferences unchanged.

---

## 5. Response Synthesizer (Answerer)

**Code location**

* File: `src/orchestration/graph.py`
* Function: `response_synthesizer`

**Role**

* Generate the short spoken recommendation that TTS reads
* Respect price budget
* Use only grounded data from:

  * `catalog_items` (private RAG results)
  * `web_items` (live web results)
* Render catalog and web cards for the UI
* Optionally append a small conflict summary if `state["price_conflicts"]` is non empty
  (that conflict text is a fixed template in code, not driven by a prompt)

**System prompt**

```text
You are the Answerer agent for a shopping assistant.
You receive structured data about products from a private catalog and from live web search.
Write a concise spoken answer (2 to 4 sentences) recommending up to 2 options.
If both sources have relevant items, briefly compare them on factors such as price and features.
Respect the user's budget if provided.
Do not invent details that are not in the data. Do not fabricate prices or ratings.
```

**User message format**

```text
Here is the data: {summary_payload as JSON}
```

Where `summary_payload` includes:

* `original_query`
* `budget`
* `catalog_items`: list of up to 3 items with title, price, brand, rating, model number, weight, and selected metadata like features and category
* `web_items`: list of up to 3 web products with title, price, availability, and source

The text generated here becomes the first paragraph of the final `response_text` and is used for TTS. The product cards and the optional conflict summary section are added by code after this LLM call.

---

## 6. Safety messages

Safety handling is intentionally simple and mostly hard coded, not prompt based.

**Router safety check**

In `router_agent`, before calling the LLM, the system:

* Lowercases the user text
* Checks it against a small banned keyword list, for example:

  * `"explosive"`, `"bomb"`, `"pesticide"`, `"poison"`, `"weapon"`, `"gunpowder"`, etc
* If any banned keyword is present, it sets:

```python
state["safety_flags"]["blocked"] = True
state["safety_flags"]["reason"] = "disallowed or dangerous product category"
state["intent"] = "blocked"
state["last_query"] = user_text
state["response_text"] = (
    "I can’t help with that request. Please ask about safe, legal household items instead."
)
```

The graph then routes directly to the Response Synthesizer, which simply returns this fixed `response_text` to the UI and TTS. No tools are called and no additional LLM prompts are used.

There is no separate safety generation prompt. Safety behavior is controlled by this allowlist based guard and fixed messages.

---

## 7. MCP tool side notes

The MCP server itself (`src/mcp/mcp_server.py`) does not use LLM prompts. It simply:

* Accepts JSON commands over stdio:

  * `{"cmd": "discover"}`
  * `{"cmd": "call", "tool": "...", "payload": {...}}`
* Routes them to:

  * `rag_search` in `src/mcp/rag_tool.py`
  * `web_search` in `src/mcp/web_tool.py`

The LLM side instructions about how to use these tools are all in:

* Planner Agent prompt
* Product Discovery Agent prompt

which are already documented above.

---