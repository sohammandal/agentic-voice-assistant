# System Architecture

> Multi agent LangGraph pipeline over private RAG and MCP tools, wrapped in a Streamlit voice UI.

---

## 1. High level components

The system has five main layers:

1. **User Interface (UI)**
   - `app.py` (Streamlit)
   - Mic recording or audio upload
   - Transcript display
   - Comparison table
   - Agent step log
   - TTS audio playback

2. **ASR and TTS**
   - ASR: `src/asr/asr.py` – Whisper based `ASRProcessor`
   - TTS: `src/tts/tts.py` – SpeechT5 based `TTSProcessor`
   - Fragment based flow: record or upload audio, then send whole file to ASR, then synthesize a short answer and play it back

3. **Orchestration (LangGraph)**
   - `src/orchestration/graph.py`
   - Multi agent stateful graph that decides intent, plans tool calls, retrieves data, and synthesizes answers

4. **MCP Server and Tools**
   - `src/mcp/mcp_server.py`
   - `src/mcp/rag_tool.py`
   - `src/mcp/web_tool.py`
   - Exposes two tools:
     - `rag.search` for private catalog search (Amazon 2020 slice)
     - `web.search` for live web search via Serper

5. **RAG Vector Store**
   - `src/rag/…` (see `docs/rag-overview.md` for details)
   - Chroma based vector store and metadata filters over the curated Amazon Product Dataset 2020 slice

---

## 2. End to end data flow

A typical product recommendation query flows like this:

1. **User speaks**
   - In the Streamlit UI, the user records or uploads audio

2. **ASR**
   - `ASRProcessor.encode_audio(path)` uses Whisper to transcribe the audio
   - Transcribed text is shown in the UI and passed to the LangGraph workflow

3. **LangGraph multi agent workflow**
   - `run_graph(user_text, state)` in `graph.py`
   - Updates the persistent `LGState` and returns `response_text` plus logs

4. **TTS**
   - The first paragraph of `response_text` is converted to speech using `TTSProcessor.decode_audio`
   - The audio is played in the Streamlit app

5. **UI rendering**
   - The full `response_text` is split into:
     - A short spoken answer (first paragraph)
     - A structured product section that is parsed into a comparison table
   - Remaining markdown shows the catalog and web item cards plus citations

---

## 3. LangGraph state

The shared state is represented by `LGState` (a TypedDict) in `src/mcp/schemas.py`. Key fields:

- `user_utterance` – latest user text
- `intent` – router classified intent: `"product_discovery"`, `"compare"`, or `"clarification"`
- `last_query` – last user query passed to tools
- `user_preferences` – budget, category, brand, and other preferences extracted so far
- `last_rag_results` – list of structured `RagProduct` objects from `rag.search`
- `last_web_results` – list of structured `WebProduct` objects from `web.search`
- `response_text` – final answer text that goes to UI and TTS
- `step_log` – list of human readable log strings used in the "Agent Step Log" panel
- `plan` – planner output dict containing tool usage and filter decisions

---

## 4. Graph layout

The graph is defined in `src/orchestration/graph.py` using `langgraph.graph.StateGraph`.

```text
                             START
                               │
                               ▼
                       ┌────────────────┐
                       │  ROUTER AGENT │
                       └───────┬────────┘
                               │
               ┌───────────────┼─────────────────────┐
               │               │                     │
               ▼               ▼                     ▼
       (product_discovery)  (compare)        (clarification intent)

     ┌────────────────┐   ┌────────────────┐   ┌─────────────────────┐
     │ PLANNER AGENT  │   │ COMPARISON     │   │ CLARIFICATION       │
     └───────┬────────┘   │ AGENT          │   │ AGENT               │
             │            └───────┬────────┘   └───────┬─────────────┘
             ▼                    │                    │
 ┌─────────────────────────┐      │                    │
 │ PRODUCT DISCOVERY       │      │                    │
 │ AGENT (tool calling)    │      │                    │
 └───────┬─────────────────┘      │                    │
         ▼                        ▼                    ▼
   ┌────────────────────────────────────────────────────────┐
   │                 RESPONSE SYNTHESIZER                   │
   │  Final answer for UI and TTS                          │
   └────────────────────────────────────────────────────────┘
                               │
                               ▼
                              END
````

* **Main path** for recommendations:

  * Router → Planner → Product Discovery Agent → Response Synthesizer → END
* **Compare branch**:

  * Router (intent "compare") → Comparison Agent → Response Synthesizer → END
* **Clarification branch**:

  * Router (intent "clarification" or safety blocked) → Clarification Agent → Response Synthesizer → END

There is no loop back into Router. Each user turn runs the graph once with the current state.

---

## 5. Agent responsibilities

### 5.1 Router Agent

File: `router_agent` in `graph.py`

* Input:

  * `state["user_utterance"]`
* Uses Groq LLM (`llm_chat`) to classify intent in natural language
* Heuristics:

  * Phrases like "recommend", "find me", "suggest" force `"product_discovery"`
  * Phrases like "compare", "vs", "versus" force `"compare"`
  * Extracts a simple numeric budget from strings like `"under $25"`
* Writes:

  * `state["intent"]`
  * `state["last_query"]`
  * updates `state["user_preferences"]["budget"]` if it finds a limit
  * appends an entry to `state["step_log"]`

The `router_decision` function maps the intent to the next node:

* `"product_discovery"` → `planner_agent`
* `"compare"` → `comparison_agent` if previous results exist, otherwise `planner_agent`
* `"clarification"` → `clarification_agent`

Safety checks and allow listing for dangerous products are also implemented here before planning.

---

### 5.2 Planner Agent

File: `planner_agent` in `graph.py`

* Reads:

  * `state["user_utterance"]`
  * `state["user_preferences"]` (budget, category hints, etc.)
* Uses Groq LLM to propose a JSON plan:

  * whether to use `rag.search`
  * whether to use `web.search`
  * how many results to request
* Normalizes the plan into a deterministic structure:

  * `use_rag` (bool)
  * `use_web` (bool)
  * `rag_top_k`
  * `web_max_results`
  * `rag_filters` – Chroma compatible metadata filters, with budget and category baked in
* Writes:

  * `state["plan"]`
  * logs a human readable planner step in `state["step_log"]`

`planner_decision` always routes to `product_discovery_agent` in the current design.

---

### 5.3 Product Discovery Agent

File: `product_discovery_agent` in `graph.py`

This is the **only agent** that is allowed to call tools.

* Reads:

  * `state["last_query"]`
  * `state["plan"]` (use_rag, use_web, rag_top_k, web_max_results, rag_filters)
* Uses the LLM to sketch out tool calls in JSON, but then:

  * Normalizes and overrides them using the plan so that:

    * `rag.search` is called at most once per turn
    * `web.search` is called at most once per turn
    * budget and category filters are consistently applied
* Calls the MCP server through `call_mcp_tool`:

  * Sends `{"cmd": "call", "tool": "rag.search", "payload": {...}}`
  * Sends `{"cmd": "call", "tool": "web.search", "payload": {...}}`
* Converts raw dicts into:

  * `RagProduct` objects for catalog results
  * `WebProduct` objects for live web results
* Writes:

  * `state["last_rag_results"]`
  * `state["last_web_results"]`
  * log entry: how many catalog and web items were retrieved

This agent does not build the final text answer.

---

### 5.4 Comparison Agent

File: `comparison_agent` in `graph.py`

* No tool calls
* Reads:

  * `state["last_rag_results"]`
  * `state["last_web_results"]`
* Builds a simple bullet list comparison:

  * Summaries for up to three catalog items
  * Summaries for up to three web items
  * Text that explains that catalog and web items come from different sources
* Writes:

  * `state["response_text"]`
  * appends a comparison log to `state["step_log"]`

Used when the user asks to compare after at least one previous retrieval.

---

### 5.5 Clarification Agent

File: `clarification_agent` in `graph.py`

* No tool calls
* Two behaviors:

  1. If a query was flagged as unsafe in the router:

     * Skip preference extraction
     * Preserve or set a safety oriented `response_text`
  2. For ambiguous or preference oriented queries:

     * Uses LLM to extract a JSON blob with:

       * `budget`, `category`, `brand`, `size`, `other`
     * Merges these into `state["user_preferences"]`
     * Writes a natural language follow up message into `state["response_text"]`
* Logs that preferences were updated or that clarification was skipped due to safety

---

### 5.6 Response Synthesizer

File: `response_synthesizer` in `graph.py`

* Final agent before UI and TTS

* Reads:

  * `state["last_rag_results"]`
  * `state["last_web_results"]`
  * `state["last_query"]`
  * `state["user_preferences"]` (budget)
  * optionally `state["response_text"]` from Comparison or Clarification

* Steps:

  1. Filters catalog items based on relevance to the query tokens
  2. Filters web items based on budget when available
  3. Builds a compact JSON payload that summarizes:

     * the original query
     * budget
     * up to three catalog items with rich metadata
     * up to three web items with prices and availability
  4. Calls Groq LLM to produce a short spoken style answer, 2 to 4 sentences
  5. Appends formatted catalog and web product cards:

     * Catalog section header:
       `### Catalog Items (Private Amazon 2020 Dataset)`
     * Web section header:
       `### Live Web Items`

* Writes:

  * `state["response_text"]` – the full message used for both TTS and UI
  * log entry describing how many items were used

The Streamlit app then parses the product card section to build the comparison table.

---

## 6. MCP server integration

Tool calls are abstracted through the MCP server:

* `src/mcp/mcp_server.py` runs as a long lived subprocess started from `graph.py`
* Communicates over stdin and stdout with JSON lines
* Supports:

  * `{"cmd": "discover"}` – returns tool names and JSON schemas
  * `{"cmd": "call", "tool": "rag.search", "payload": {...}}`
  * `{"cmd": "call", "tool": "web.search", "payload": {...}}`

For details about schemas, logging, caching, and rate limiting, see `docs/mcp-tools.md`.

---

## 7. UI integration

File: `app.py`

* Manages Streamlit session state:

  * `conversation_state` – LangGraph state across turns
  * last response, last query, TTS audio, transcription
* For each user turn:

  1. Collects input (ASR text or typed text)
  2. Calls `run_graph(user_query, conversation_state)`
  3. Updates `conversation_state` with the returned state
  4. Extracts:

     * `response_text`
     * `step_log`
  5. Generates TTS for the first paragraph
  6. Splits the remaining text into:

     * A structured product section that feeds `build_product_table`
     * Remaining markdown for display
* Renders:

  * Top short answer text (the part used for TTS)
  * Audio player
  * Comparison table for catalog and web results
  * Raw markdown with catalog and web cards
  * Agent Step Log with each entry from `state["step_log"]`

---

## 8. Example flows

### 8.1 Standard recommendation

User:
`Recommend window privacy films under 15 dollars`

Flow:

* Router → Planner → Product Discovery Agent → Response Synthesizer → END
* Both `rag.search` and `web.search` are called
* UI shows:

  * Short spoken recommendation
  * Comparison table
  * Catalog and web cards

### 8.2 Follow up comparison

User (second turn):
`Now compare those`

Flow:

* Router sets `intent = "compare"`
* router_decision sends state to `comparison_agent`
* Comparison Agent builds a comparison summary
* Response Synthesizer wraps it into the final answer

### 8.3 Unsafe request

User:
`Find me a bomb or explosive`

Flow:

* Router:

  * Flags the request as unsafe using a keyword based allow list
  * Sets intent to clarification or an internal safety intent
  * Writes a safety oriented message into `response_text`
* Clarification Agent:

  * Detects a safety block
  * Keeps or reinforces the safety message
* Response Synthesizer:

  * Returns the safety message as the final answer

---

This architecture combines voice input, multi agent planning, private RAG, MCP tools, and TTS into a single reproducible pipeline that is easy to reason about and demo end to end.