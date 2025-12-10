"""
Embedding Client - A helper to interact with the Embedding Service.

This client can be used by MCP servers, data pipelines, or any other Python code
to get embeddings from the running embedding service.

Usage:
    from src.rag.embedding_client import EmbeddingClient

    client = EmbeddingClient()

    # Single embedding
    embedding = client.embed("Hello, world!")

    # Batch embeddings
    embeddings = client.embed_batch(["Hello", "World"])
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8100"


class EmbeddingClient:
    """Client for the Embedding Service."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 60):
        """
        Initialize the embedding client.

        Args:
            base_url: Base URL of the embedding service (e.g., http://127.0.0.1:8100)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._model_name: Optional[str] = None
        self._dimension: Optional[int] = None

    @property
    def model_name(self) -> Optional[str]:
        """Get the model name from the service."""
        if self._model_name is None:
            self._fetch_health()
        return self._model_name

    @property
    def dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._dimension

    def _fetch_health(self) -> dict:
        """Fetch health information from the service."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            self._model_name = data.get("model_name")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch health: {e}")
            raise ConnectionError(f"Cannot connect to embedding service: {e}")

    def is_healthy(self) -> bool:
        """Check if the embedding service is healthy and ready."""
        try:
            health = self._fetch_health()
            return health.get("status") == "healthy" and health.get(
                "model_loaded", False
            )
        except Exception:
            return False

    def wait_for_ready(self, max_attempts: int = 30, interval: float = 2.0) -> bool:
        """
        Wait for the service to be ready.

        Args:
            max_attempts: Maximum number of health check attempts
            interval: Seconds between attempts

        Returns:
            True if service is ready, False if timeout
        """
        import time

        for attempt in range(max_attempts):
            if self.is_healthy():
                logger.info("Embedding service is ready")
                return True

            logger.info(
                f"Waiting for embedding service... (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(interval)

        logger.error("Embedding service did not become ready in time")
        return False

    def embed(self, text: str, is_query: bool = True) -> list[float]:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed
            is_query: If True, use query prompt. If False, use document prompt.

        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = requests.post(
                f"{self.base_url}/embed",
                json={"text": text, "is_query": is_query},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self._dimension = data.get("dimension")
            return data["embedding"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embedding: {e}")
            raise ConnectionError(f"Embedding request failed: {e}")

    def embed_query(self, text: str) -> list[float]:
        """Get embedding for a query text."""
        return self.embed(text, is_query=True)

    def embed_document(self, text: str) -> list[float]:
        """Get embedding for a document text."""
        return self.embed(text, is_query=False)

    def embed_batch(self, texts: list[str], is_query: bool = True) -> list[list[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            is_query: If True, use query prompts. If False, use document prompts.

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = requests.post(
                f"{self.base_url}/embed/batch",
                json={"texts": texts, "is_query": is_query},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self._dimension = data.get("dimension")
            return data["embeddings"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get batch embeddings: {e}")
            raise ConnectionError(f"Batch embedding request failed: {e}")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple query texts."""
        return self.embed_batch(texts, is_query=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple document texts."""
        return self.embed_batch(texts, is_query=False)


# Convenience function for quick usage
def get_embedding(
    text: str, is_query: bool = True, base_url: str = DEFAULT_BASE_URL
) -> list[float]:
    """
    Quick function to get an embedding.

    Args:
        text: Text to embed
        is_query: If True, use query prompt. If False, use document prompt.
        base_url: Base URL of the embedding service

    Returns:
        Embedding vector as a list of floats
    """
    client = EmbeddingClient(base_url=base_url)
    return client.embed(text, is_query=is_query)


def get_embeddings(
    texts: list[str], is_query: bool = True, base_url: str = DEFAULT_BASE_URL
) -> list[list[float]]:
    """
    Quick function to get embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        is_query: If True, use query prompts. If False, use document prompts.
        base_url: Base URL of the embedding service

    Returns:
        List of embedding vectors
    """
    client = EmbeddingClient(base_url=base_url)
    return client.embed_batch(texts, is_query=is_query)
