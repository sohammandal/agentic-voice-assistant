# Prompts

This document collects all the main prompts used by the system, grouped by agent and tool.

It is meant to satisfy the Prompt Disclosure requirement in the grading rubric:
- System prompts
- Router and planner tool prompts
- JSON format instructions
- Safety related instructions
- How prompts map to code and agents

The model is configured in:

- `src/config.py`  
  - `GROQ_MODEL_NAME = "llama-3.3-70b-versatile"`  
  - `LLM_TEMPERATURE` and `LLM_MAX_TOKENS`  
- `src/llm/groq_llm.py`  
  - `llm_chat` wraps the Groq client

Each section below shows the **system prompt** and how it is used.

---

## 1. Router Agent

**Code location**  
- File: `src/orchestration/graph.py`  
- Function: `router_agent`  

**Role**  
- Classifies the user intent into:
  - `product_discovery`
  - `compare`
  - `clarification`
- Also seeds budget into `user_preferences` via regex

**System prompt**

```text
You are the Router Agent.
Classify the user's intent using only the text.
Choose one of:
- product_discovery  (user wants help finding products)
- compare            (user wants comparison of items already found)
- clarification      (user unsure, talking about budget/preferences)
Respond in natural language. Do NOT call tools.
````

**User message format**

```text
{user_utterance}
```

Heuristics in code then override this classification for phrases like:

* "recommend", "suggest", "find me", "show me" ⇒ force `product_discovery`
* "compare", " vs ", " versus " ⇒ force `compare`

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

The JSON returned by the model is parsed and merged into a default plan, then further adjusted by code to add actual Chroma filters.

---

## 3. Product Discovery Agent (tool calling)

**Code location**

* File: `src/orchestration/graph.py`
* Function: `product_discovery_agent`

**Role**

* This is the **only** agent allowed to call MCP tools.
* It uses the LLM to suggest tool calls, then the code normalizes them and enforces the planned settings (top_k, filters, etc).
* Calls:

  * `rag.search` for private catalog
  * `web.search` for live web pricing and availability

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
* Falls back to a deterministic plan based tool call if JSON parsing fails

---

## 4. Clarification Agent

**Code location**

* File: `src/orchestration/graph.py`
* Function: `clarification_agent`

**Role**

* Extracts user preferences when the intent is `clarification`
* If a safety flag is set (unsafe query), it **skips** preference extraction and preserves the safety response instead

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

If YAML or JSON parsing fails, the code falls back to a default structure with nulls.

For safety blocked queries, the agent does not call this prompt and instead returns a fixed safety message.

---

## 5. Response Synthesizer (Answerer)

**Code location**

* File: `src/orchestration/graph.py`
* Function: `response_synthesizer`

**Role**

* Generate the **short spoken recommendation** that TTS reads
* Respect price budget
* Use only grounded data from:

  * `catalog_items` (private RAG results)
  * `web_items` (live web results)

**System prompt**

```text
You are the Answerer agent for a shopping assistant.
You receive structured data about products from a private catalog and from live web search.
Write a concise spoken answer (2 to 4 sentences) recommending up to 2 options.
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
* `catalog_items`: list of up to 3 items with title, price, brand, rating, features, model number, etc
* `web_items`: list of up to 3 web products with title, price, availability, and source

The text generated here becomes the first paragraph of the final `response_text` and is used for TTS.

---

## 6. Safety messages

In addition to the prompts above, the system uses fixed string messages for unsafe requests and does not rely on the LLM to decide safety at generation time.

**Examples of hard coded safety responses**

* In the Clarification agent when a query is blocked:

```text
I am not able to help with potentially dangerous or restricted products.
I can help you find everyday household items and common cleaning products instead.
```

---

## 7. MCP tool side notes

The MCP server itself (`src/mcp/mcp_server.py`) does not use LLM prompts. It simply:

* Accepts JSON commands over stdio:

  * `{"cmd": "discover"}`
  * `{"cmd": "call", "tool": "...", "payload": {...}}`
* Routes them to:

  * `rag_search` in `src/mcp/rag_tool.py`
  * `web_search` in `src/mcp/web_tool.py`

The **LLM side instructions** about how to use these tools are all in:

* Planner Agent prompt
* Product Discovery Agent prompt

which are already documented above.

---