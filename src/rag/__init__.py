"""
RAG (Retrieval-Augmented Generation) module.

This module provides:
- embedding_service: A standalone HTTP REST API service for text embeddings
- embedding_client: A client to interact with the embedding service
- run_service: Utilities to start/stop the embedding service programmatically

Usage:
    # Use the client to get embeddings
    from src.rag.embedding_client import EmbeddingClient

    client = EmbeddingClient()
    embedding = client.embed("Hello, world!")
    embeddings = client.embed_batch(["Hello", "World"])

    # For programmatic service management, import directly:
    from src.rag.run_service import EmbeddingServiceManager

    manager = EmbeddingServiceManager()
    manager.start()
    manager.stop()
"""

from src.rag.embedding_client import EmbeddingClient, get_embedding, get_embeddings

__all__ = [
    "EmbeddingClient",
    "get_embedding",
    "get_embeddings",
]
