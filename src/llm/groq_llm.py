# This file provides a clean wrapper around the Groq OpenAI-compatible API, using your selected model

# This wrapper is used by:
# Router Agent (optional)
# Clarification Agent
# Comparison Agent
# Response Synthesizer (when needed)

# This module ensures every agent call is:
# consistent
# short
# simple to use

"""
groq_llm.py

Wrapper for calling the Groq Llama3-70B model through the
OpenAI-compatible chat.completions endpoint.

This provides a simple `llm_chat()` function used by LangGraph agents.
"""

import os
from typing import Any, Dict, List

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL_NAME, LLM_MAX_TOKENS, LLM_TEMPERATURE

# ------------------------------------------------------------------------------
# Initialize Groq Client
# ------------------------------------------------------------------------------
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

client = Groq(api_key=GROQ_API_KEY)


# ------------------------------------------------------------------------------
# Main Chat Function
# ------------------------------------------------------------------------------
def llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """
    Send a chat completion to the Groq LLM (Llama3-70B).

    Args:
        messages: list of {"role": "user" | "assistant" | "system", "content": "..."}
        temperature: sampling temperature
        max_tokens: max tokens in output

    Returns:
        assistant message text string
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message["content"]
