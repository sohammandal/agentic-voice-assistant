"""
config.py

Central configuration for the Agentic Voice Assistant project.
This file provides paths, environment variables, and global constants
used across tools, MCP server, LangGraph, and utility modules.
"""

import os
from pathlib import Path

# ------------------------------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------------------------------

# Root directory = <repo_root>/src/... so use parent of current file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location of your Chroma persistent directory (exact from your repo)
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# The Chroma collection name you provided
CHROMA_COLLECTION_NAME = "agentic_voice_assistant_vdb"

# ------------------------------------------------------------------------------------
# Web Search (Serper)
# ------------------------------------------------------------------------------------
# SERPER API key — must be set in your .env or system environment
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# Cache time: 5 minutes
WEB_SEARCH_TTL_SECONDS = 300

# Simple rate limit for external API calls (seconds)
WEB_SEARCH_MIN_INTERVAL = 1.0

# ------------------------------------------------------------------------------------
# Groq LLM Settings
# ------------------------------------------------------------------------------------

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Your selected model
GROQ_MODEL_NAME = "llama3-70b-8192"

# Default temperature / max tokens for agent reasoning
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# ------------------------------------------------------------------------------------
# Agent Defaults
# ------------------------------------------------------------------------------------

DEFAULT_TOP_K_RAG = 10
DEFAULT_MAX_WEB_RESULTS = 5

# The maximum number of products shown in ResponseSynthesizer for readability
MAX_PRODUCTS_SHOWN = 3

# ------------------------------------------------------------------------------------
# ASR & TTS (if you later integrate external APIs)
# ------------------------------------------------------------------------------------
# Placeholders for future use
ASR_PROVIDER = "placeholder"
TTS_PROVIDER = "placeholder"

# ------------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------------


def validate_env():
    """Optional: call this to ensure API keys are loaded correctly."""
    missing = []
    if not SERPER_API_KEY:
        missing.append("SERPER_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if missing:
        print("WARNING: Missing environment variables for:", ", ".join(missing))
