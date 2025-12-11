                                  ┌──────────────────────────────┐
                                  │            START             │
                                  └───────────────┬──────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │         ROUTER AGENT           │
                                  │  Reads user_utterance          │
                                  │  Classifies intent             │
                                  │  Writes: state.intent          │
                                  │          state.last_query      │
                                  └───────────────┬────────────────┘
                                                  │
                          ┌───────────────────────┼───────────────────────────────┐
                          │                       │                               │
                          ▼                       ▼                               ▼
        (if intent="product_discovery")   (if intent="compare" AND       (if intent="clarification")
                                          results already exist)

┌────────────────────────────┐    ┌──────────────────────────┐    ┌───────────────────────────┐
│ PRODUCT DISCOVERY AGENT    │    │    COMPARISON AGENT      │    │    CLARIFICATION AGENT    │
│ (ONLY agent that calls     │    │  (NO tool calls)         │    │   (NO tool calls)         │
│   rag.search + web.search) │    │  Uses existing results   │    │   Extracts preferences    │
│                            │    │  → writes response_text  │    │   Updates preferences     │
│ Writes:                    │    └───────────────┬──────────┘    │   Asks follow-up question │
│   state.last_rag_results   │                    │               └───────────────┬───────────┘
│   state.last_web_results   │                    │                               │
└───────────────┬────────────┘                    │                               │
                │                                 │                               │
                ▼                                 ▼                               │
      ┌────────────────────────────┐     ┌────────────────────────────┐           │
      │    RESPONSE SYNTHESIZER    │     │    RESPONSE SYNTHESIZER    │           │
      │  Builds final response     │     │  Builds final response     │           │
      └───────────────┬────────────┘     └───────────────┬────────────┘           │
                      │                                  │                        │
                      ▼                                  ▼                        │
              ┌──────────────┐                  ┌──────────────┐                  │
              │     END      │                  │     END      │                  │
              └──────────────┘                  └──────────────┘                  │
                                                                                  │
                                                                                  ▼
                                                                   (Clarification ALWAYS loops back)
                                                                 ┌──────────────────────────────────┐
                                                                 │          ROUTER AGENT           │
                                                                 └──────────────────────────────────┘
