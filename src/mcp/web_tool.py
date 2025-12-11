# a Serper-style implementation

# Your Live Web Search Tool (Serper API + Caching + Rate Limit).
# This file implements the web.search MCP tool exactly as required:
# Calls Serper (real web search)
# Extracts title, url, snippet
# Extracts price, availability when possible
# Adds caching (TTL = 300s)
# Adds rate limiting (1 request/sec)
# Returns structured objects using your WebProduct schema

# web_tool.py
"""
web_tool.py

Implements the web.search tool using Serper's Web Search API.
Includes:
- Rate limiting
- Caching (TTL 300s)
- Output normalized into WebProduct schema
"""

import time
from typing import Dict, List, Optional

import requests

from src.config import (
    DEFAULT_MAX_WEB_RESULTS,
    SERPER_API_KEY,
    WEB_SEARCH_MIN_INTERVAL,
    WEB_SEARCH_TTL_SECONDS,
)

from .schemas import WebProduct

# --------------------------------------------------------------------------
# Safety checks
# --------------------------------------------------------------------------
if not SERPER_API_KEY:
    raise ValueError("SERPER_API_KEY is missing from environment variables.")

SERPER_URL = "https://google.serper.dev/search"

_last_call_timestamp = 0.0
_serper_cache: Dict[str, Dict] = {}
_serper_cache_timestamps: Dict[str, float] = {}


# --------------------------------------------------------------------------
# Rate Limiter
# --------------------------------------------------------------------------
def _rate_limit():
    """Ensure we do not call Serper too frequently."""
    global _last_call_timestamp
    now = time.time()
    elapsed = now - _last_call_timestamp

    if elapsed < WEB_SEARCH_MIN_INTERVAL:
        time.sleep(WEB_SEARCH_MIN_INTERVAL - elapsed)

    _last_call_timestamp = time.time()


# --------------------------------------------------------------------------
# LRU Cache for responses
# --------------------------------------------------------------------------
def _cached_serper_request(query: str) -> Dict:
    """Serper request with simple in memory TTL cache."""
    now = time.time()

    # Return cached result if still fresh
    if query in _serper_cache:
        ts = _serper_cache_timestamps.get(query, 0.0)
        if now - ts < WEB_SEARCH_TTL_SECONDS:
            return _serper_cache[query]

    # Otherwise call Serper
    _rate_limit()

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}

    resp = requests.post(SERPER_URL, json=payload, headers=headers, timeout=10)

    if resp.status_code != 200:
        data = {"organic": []}
    else:
        data = resp.json()

    # Store in cache with timestamp
    _serper_cache[query] = data
    _serper_cache_timestamps[query] = now

    return data


# --------------------------------------------------------------------------
# Helper to extract price (very heuristic)
# --------------------------------------------------------------------------
def _extract_price(snippet: str) -> Optional[float]:
    """
    Try to extract a price from snippet text.
    Extremely naive — can be replaced with regex or parsing.
    """
    import re

    matches = re.findall(r"\$\d+(?:\.\d+)?", snippet)
    if matches:
        try:
            return float(matches[0].replace("$", ""))
        except:
            return None
    return None


# --------------------------------------------------------------------------
# Convert Serper JSON → WebProduct objects
# --------------------------------------------------------------------------
def _convert_item_to_web_product(item: Dict) -> WebProduct:
    title = item.get("title", "")
    url = item.get("link", "")
    snippet = item.get("snippet", "")

    price = _extract_price(snippet)

    # Availability extraction (very naive)
    availability = None
    if "in stock" in snippet.lower():
        availability = "In Stock"
    elif "out of stock" in snippet.lower():
        availability = "Out of Stock"

    return WebProduct(
        title=title,
        url=url,
        snippet=snippet,
        price=price,
        availability=availability,
        source="serper",
        raw=item,
    )


# --------------------------------------------------------------------------
# Main web.search function
# --------------------------------------------------------------------------
def web_search(
    query: str, max_results: int = DEFAULT_MAX_WEB_RESULTS
) -> List[WebProduct]:
    """
    Perform Serper web search and return a list of WebProduct objects.
    Results are cached and rate-limited automatically.
    """

    data = _cached_serper_request(query)

    organic = data.get("organic", [])[:max_results]

    out: List[WebProduct] = []

    for item in organic:
        out.append(_convert_item_to_web_product(item))

    return out
