# This file defines all typed schemas for:
# RAG Products
# Web Search Products
# Conversation State for LangGraph
# Internal structures used across tools

# schemas.py
"""
schemas.py

Structured dataclasses and TypedDicts for:
- RAG product outputs
- Web product outputs
- Conversation state
These are used across tools, MCP server, and LangGraph orchestration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------------------
# RAG Product Schema (Private Catalog)
# ------------------------------------------------------------------------------
@dataclass
class RagProduct:
    """
    Represents a product returned from rag.search (private vector DB).
    This is mapped directly from your metadata stored in Chroma.
    """

    # sku: str  # product_id
    # title: str  # product_name
    # price: Optional[float]  # price
    # rating: Optional[float]  # None if missing
    # brand: Optional[str]  # brand
    # ingredients: Optional[str]  # ingredient text or None
    # doc_id: str  # Chroma internal chunk id (text_xxx_1)
    # shipping_weight_lbs: Optional[float]  # shipping weight
    # model_number: Optional[str]  # model_number
    # raw_metadata: Dict[str, Any] = field(default_factory=dict)

    # Required (always in embedding_text):
    sku: str  # product_id (used for doc_id)
    title: str  # product_name (first line of embedding_text)
    doc_id: str  # ChromaDB chunk ID
    price: Optional[float] = None  # "Price per pound: $X.XX"
    brand: Optional[str] = None  # "Brand: ..." line
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    document_text: Optional[str] = None  # The full embedding_text

    # Optional (NOT in embedding_text - remove from metadata):
    rating: Optional[float] = None  # Not in embedding_text
    ingredients: Optional[str] = None  # Not in embedding_text
    model_number: Optional[str] = None  # Not in embedding_text
    shipping_weight_lbs: Optional[float] = None  # Not in embedding_text


# ------------------------------------------------------------------------------
# Web Product Schema (Live Web Search Results)
# ------------------------------------------------------------------------------
@dataclass
class WebProduct:
    """
    Represents a product returned from web.search (Serper).
    """

    title: str
    url: str
    snippet: str
    price: Optional[float]
    availability: Optional[str]
    source: str  # "serper"
    raw: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Conversation State (LangGraph)
# ------------------------------------------------------------------------------
@dataclass
class ConversationState:
    """
    Unified state container passed between LangGraph agents.
    This mirrors the diagram:
        - last_query
        - user_preferences
        - last_rag_results
        - last_web_results
    And also includes internal keys:
        - intent
        - user_utterance
        - response_text
    """

    user_utterance: Optional[str] = None
    intent: Optional[str] = None

    last_query: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    last_rag_results: List[RagProduct] = field(default_factory=list)
    last_web_results: List[WebProduct] = field(default_factory=list)

    response_text: Optional[str] = None


# ------------------------------------------------------------------------------
# TypedDict version for LangGraph (some prefer this over dataclasses)
# ------------------------------------------------------------------------------
from typing import TypedDict


class LGState(TypedDict, total=False):
    """
    LangGraph-compatible dictionary state.
    Same fields as ConversationState, but JSON-friendly.
    """

    user_utterance: str
    intent: str
    last_query: str

    user_preferences: Dict[str, Any]
    last_rag_results: List[Dict[str, Any]]
    last_web_results: List[Dict[str, Any]]

    response_text: str
    step_log: List[str]
    plan: Dict[str, Any]
