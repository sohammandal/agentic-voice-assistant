# src/mcp/rag_tool.py

"""
rag_tool.py

Implements rag.search on top of your persistent Chroma vector store, via
VectorStoreService:

- Loads existing collection (CHROMA_COLLECTION_NAME)
- Performs semantic search with optional metadata filters
- Maps Chroma metadata into structured RagProduct objects
"""

from typing import Any, Dict, List, Optional

from src.mcp.schemas import RagProduct
from src.rag.vector_store_service import VectorStoreService

_vector_store: Optional[VectorStoreService] = None


def _get_vector_store() -> VectorStoreService:
    """Lazy-init a single VectorStoreService instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store


def rag_search(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RagProduct]:
    """
    RAG search over the prebuilt Chroma vector store.

    Args:
        query: User query text.
        top_k: Number of results to return.
        filters: Optional metadata filters, passed as `where` to VectorStoreService.search.

    Returns:
        List[RagProduct] for LangGraph / MCP.
    """
    vs = _get_vector_store()

    results = vs.search(
        query=query,
        top_k=top_k,
        where=filters,
    )

    rag_products: List[RagProduct] = []
    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    dists = results.get("distances", [])

    for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
        rag_products.append(
            RagProduct(
                sku=str(meta.get("product_id", doc_id)),
                title=str(meta.get("product_name", "Unknown product")),
                price=meta.get("price"),
                rating=None,  # your slice has no rating column
                brand=meta.get("brand"),
                ingredients=None,
                doc_id=str(doc_id),
                shipping_weight_lbs=meta.get("shipping_weight_lbs"),
                model_number=meta.get("model_number"),
                raw_metadata=meta,
            )
        )

    return rag_products
