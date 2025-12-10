# RAG: rag.search over your Chroma vector store (rag_tool.py)

# This assumes:
# You already built a persistent Chroma store under vector_store/

# Each record has:
# id: chunk id (e.g. text_<product_id>_1)
# document: your embedding_text
# metadata: JSON that includes things like product_id, product_name, selling_price, brand_name, etc.

# Purpose: encapsulates everything about private catalog retrieval so:
# MCP → rag_search()
# LangGraph agents → only see clean RagProduct objects

# Your RAG Tool, fully mapped to your real Chroma metadata schema.
# This file:
# Loads  existing Chroma persistent store
# Connects to collection: agentic_voice_assistant_vdb
# Handles similarity search
# Converts  real metadata to the rag.search output schema approved

# Router Agent → Product Discovery Agent
# The Product Discovery Agent receives this structured list:
# state.last_rag_results = [RagProduct(...)]
# Then the agent:
# Displays catalog price (12.99)
# Displays brand, title
# Shows a product card
# And also performs web.search separately
# This follows your "NO FUSION" rule.

# rag_tool.py
"""
rag_tool.py

Implements rag.search on top of your persistent Chroma vector store:
- Loads existing collection: agentic_voice_assistant_vdb
- Performs similarity search
- Maps metadata into structured RagProduct objects
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_COLLECTION_NAME, DEFAULT_TOP_K_RAG, VECTOR_STORE_DIR

from .schemas import RagProduct

# ------------------------------------------------------------------------------
# Connect to Chroma persistent DB
# ------------------------------------------------------------------------------
# Your folder structure:
# vector_store/
#   aa7e9642-b045-47d3-8062-92bd59a7d5da/
#       chroma.sqlite3, *.bin files

_client = chromadb.PersistentClient(
    path=str(VECTOR_STORE_DIR),
    settings=Settings(anonymized_telemetry=False),
)

# Load collection by name (you confirmed this)
_collection = _client.get_collection(CHROMA_COLLECTION_NAME)


# ------------------------------------------------------------------------------
# Metadata → RagProduct mapper
# ------------------------------------------------------------------------------
def _metadata_to_rag_product(meta: Dict[str, Any], doc_id: str) -> RagProduct:
    """
    Convert your actual metadata dictionary from Chroma into RagProduct format.
    """

    # # --- Required fields from your dataset ---
    # product_id = meta.get("product_id")
    # product_name = meta.get("product_name")
    # price = meta.get("price")

    # # Some items have rating, some don't
    # rating = meta.get("rating")  # may be None

    # brand = meta.get("brand")
    # ingredients = meta.get("ingredients")

    # # Extra fields you asked for
    # shipping_weight_lbs = meta.get("shipping_weight_lbs")
    # model_number = meta.get("model_number")

    product_id = meta.get("product_id")
    product_name = meta.get("product_name")
    price = meta.get("price")
    brand = meta.get("brand")

    # return RagProduct(
    #     sku=str(product_id),
    #     title=str(product_name),
    #     price=float(price) if price is not None else None,
    #     rating=rating,
    #     brand=brand,
    #     ingredients=ingredients,
    #     doc_id=str(doc_id),
    #     shipping_weight_lbs=shipping_weight_lbs,
    #     model_number=model_number,
    #     raw_metadata=meta,
    # )
    return RagProduct(
        sku=str(product_id),
        title=str(product_name),
        price=float(price) if price is not None else None,
        rating=None,  # Always None
        brand=brand,
        ingredients=None,  # Always None
        doc_id=str(doc_id),
        shipping_weight_lbs=None,  # Not storing this
        model_number=None,  # Not storing this
        raw_metadata=meta,
        document_text=document_text,  # ← ADD THIS
    )


# ------------------------------------------------------------------------------
# Main rag.search function
# ------------------------------------------------------------------------------
def rag_search(
    query: str,
    top_k: int = DEFAULT_TOP_K_RAG,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RagProduct]:
    """
    Perform semantic search over your Chroma vector DB.
    This is the core of MCP tool `rag.search`.

    Args:
        query: natural language search query
        top_k: number of results
        filters: optional metadata filters (dict)

    Returns:
        List[RagProduct]
    """

    where = filters or None

    results = _collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["metadatas", "ids", "documents"],
    )

    ids = results.get("ids", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]

    products: List[RagProduct] = []

    for doc_id, meta, document_text in zip(ids, metas, docs):
        products.append(_metadata_to_rag_product(meta, doc_id, document_text))

    return products
