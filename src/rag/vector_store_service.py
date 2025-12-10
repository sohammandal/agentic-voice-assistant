"""
Vector Store Service using ChromaDB.

This service provides:
- Loading product data from CSV files
- Creating embeddings using the embedding service
- Storing embeddings in ChromaDB with metadata
- Semantic search with metadata filtering capabilities

Usage:
    from src.rag.vector_store_service import VectorStoreService

    # Initialize service
    service = VectorStoreService()

    # Build vector store from CSV
    service.build_from_csv("data/homes_preprocessed_data.csv")

    # Search
    results = service.search("comfortable sofa", top_k=5)

    # Search with metadata filters
    results = service.search(
        "kitchen appliance",
        top_k=5,
        where={"main_category": "home & kitchen"},
        where_document={"$contains": "stainless steel"}
    )
"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
from chromadb.config import Settings
from tqdm import tqdm

from src.config import CHROMA_COLLECTION_NAME, VECTOR_STORE_DIR
from src.rag.embedding_client import EmbeddingClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default batch size for embedding and insertion
DEFAULT_BATCH_SIZE = 100


class VectorStoreService:
    """Service for managing ChromaDB vector store with product embeddings."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        """
        Initialize the Vector Store Service.

        Args:
            persist_directory: Directory to persist ChromaDB data.
                              Defaults to VECTOR_STORE_DIR from config.
            collection_name: Name of the ChromaDB collection.
                            Defaults to CHROMA_COLLECTION_NAME from config.
            embedding_client: EmbeddingClient instance. Creates new one if not provided.
        """
        self.persist_directory = persist_directory or str(VECTOR_STORE_DIR)
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.embedding_client = embedding_client or EmbeddingClient()

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self.collection = None
        self._dimension: Optional[int] = None

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get or create the ChromaDB collection."""
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
        return self.collection

    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Clean metadata to ensure ChromaDB compatibility.

        ChromaDB only accepts: str, int, float, bool as metadata values.
        - None values are removed
        - Lists/dicts are converted to JSON strings
        - NaN/inf floats are removed
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, bool):
                cleaned[key] = value
            elif isinstance(value, (int, float)):
                # Check for NaN or inf
                if pd.isna(value) or (
                    isinstance(value, float) and not (-1e308 < value < 1e308)
                ):
                    continue
                cleaned[key] = value
            elif isinstance(value, str):
                if value.strip():  # Only include non-empty strings
                    cleaned[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to string
                cleaned[key] = str(value)
            else:
                # Try to convert to string
                try:
                    cleaned[key] = str(value)
                except Exception:
                    continue
        return cleaned

    def _parse_metadata_from_csv(self, metadata_str: str) -> dict:
        """
        Parse metadata string from CSV back to dictionary.

        The CSV stores metadata as a string representation of a dict.
        """
        # Handle empty / NaN values
        if pd.isna(metadata_str) or not metadata_str:
            return {}

        # If it's already a dict, return as-is
        if isinstance(metadata_str, dict):
            return metadata_str

        text = str(metadata_str).strip()

        # Try JSON first (handles double-quoted JSON)
        try:
            import json

            return json.loads(text)
        except Exception:
            pass

        # Replace standalone `nan` tokens with Python None for ast.literal_eval
        try:
            text_for_ast = re.sub(r"\bnan\b", "None", text, flags=re.IGNORECASE)
        except Exception:
            text_for_ast = text

        # Try ast.literal_eval on Python-literal dicts (after normalizing nan)
        try:
            return ast.literal_eval(text_for_ast)
        except Exception:
            pass

        # As a last resort, coerce to JSON-like string and try json.loads
        try:
            s = text_for_ast
            s = (
                s.replace("None", "null")
                .replace("True", "true")
                .replace("False", "false")
            )
            s = re.sub(r"\bnan\b", "null", s, flags=re.IGNORECASE)
            # Convert single quotes to double quotes when safe
            if "'" in s and '"' not in s:
                s = s.replace("'", '"')

            import json

            return json.loads(s)
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {e}")
            return {}

    def build_from_csv(
        self,
        csv_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        text_column: str = "embedding_text",
        metadata_column: str = "metadata",
        id_column: str = "product_id",
        reset_collection: bool = False,
    ) -> int:
        """
        Build vector store from a CSV file.

        Args:
            csv_path: Path to the CSV file containing product data.
            batch_size: Number of documents to process at a time.
            text_column: Column name containing the text to embed.
            metadata_column: Column name containing metadata dictionary.
            id_column: Column name containing unique document IDs.
            reset_collection: If True, delete existing collection before building.

        Returns:
            Number of documents successfully added to the vector store.
        """
        # Resolve path relative to project root
        if not os.path.isabs(csv_path):
            from src.config import PROJECT_ROOT

            csv_path = str(PROJECT_ROOT / csv_path)

        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")

        # Validate required columns
        required_cols = [text_column, metadata_column, id_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Reset collection if requested
        if reset_collection:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
                self.collection = None
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")

        collection = self._get_or_create_collection()

        # Check if embedding service is available
        if not self.embedding_client.is_healthy():
            raise RuntimeError(
                "Embedding service is not available. "
                "Please start the service first using: python -m src.rag.run_service"
            )

        total_added = 0
        failed_count = 0

        # Process in batches
        for start_idx in tqdm(
            range(0, len(df), batch_size), desc="Building vector store"
        ):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # Prepare batch data
            ids = []
            documents = []
            metadatas = []

            for _, row in batch_df.iterrows():
                doc_id = str(row[id_column])
                text = row[text_column]
                metadata_raw = row[metadata_column]

                # Skip rows with empty text
                if pd.isna(text) or not str(text).strip():
                    logger.warning(f"Skipping document {doc_id}: empty text")
                    failed_count += 1
                    continue

                # Parse and clean metadata
                metadata = self._parse_metadata_from_csv(metadata_raw)
                metadata = self._clean_metadata(metadata)

                ids.append(doc_id)
                documents.append(str(text))
                metadatas.append(metadata)

            if not documents:
                continue

            try:
                # Get embeddings for batch (use is_query=False for documents)
                embeddings = self.embedding_client.embed_documents(documents)

                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                total_added += len(ids)

            except Exception as e:
                logger.error(f"Failed to process batch {start_idx}-{end_idx}: {e}")
                failed_count += len(ids)

        logger.info(
            f"Vector store build complete. Added: {total_added}, Failed: {failed_count}"
        )
        return total_added

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> dict:
        """
        Perform semantic search with optional metadata filtering.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            where: Metadata filter conditions.
                   Example: {"main_category": "home & kitchen"}
                   Example: {"price": {"$lt": 100}}
                   Example: {"$and": [{"price": {"$gt": 50}}, {"brand": "Sony"}]}
            where_document: Document content filter.
                           Example: {"$contains": "stainless steel"}
            include: List of fields to include in results.
                    Options: ["documents", "metadatas", "embeddings", "distances"]
                    Default: ["documents", "metadatas", "distances"]

        Returns:
            Dictionary with search results containing:
            - ids: List of document IDs
            - documents: List of document texts
            - metadatas: List of metadata dictionaries
            - distances: List of distance scores (lower is better for cosine)
        """
        collection = self._get_or_create_collection()

        if include is None:
            include = ["documents", "metadatas", "distances"]

        # Get query embedding
        query_embedding = self.embedding_client.embed_query(query)

        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": include,
        }

        if where:
            query_params["where"] = where
        if where_document:
            query_params["where_document"] = where_document

        # Execute search
        results = collection.query(**query_params)

        # Flatten results (remove outer list from single query)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results.get("documents") else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else [],
        }

    def search_by_metadata(
        self,
        where: dict,
        top_k: int = 10,
        include: Optional[list[str]] = None,
    ) -> dict:
        """
        Search documents by metadata only (no semantic search).

        Args:
            where: Metadata filter conditions.
            top_k: Maximum number of results.
            include: Fields to include in results.

        Returns:
            Dictionary with matching documents.
        """
        collection = self._get_or_create_collection()

        if include is None:
            include = ["documents", "metadatas"]

        results = collection.get(
            where=where,
            limit=top_k,
            include=include,
        )

        return results

    def get_document(self, doc_id: str) -> Optional[dict]:
        """
        Get a specific document by ID.

        Args:
            doc_id: The document ID.

        Returns:
            Dictionary with document data or None if not found.
        """
        collection = self._get_or_create_collection()

        results = collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return None

        return {
            "id": results["ids"][0],
            "document": results["documents"][0] if results.get("documents") else None,
            "metadata": results["metadatas"][0] if results.get("metadatas") else None,
        }

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        collection = self._get_or_create_collection()
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        collection = self._get_or_create_collection()

        return {
            "name": collection.name,
            "count": collection.count(),
            "metadata": collection.metadata,
        }

    def reset(self) -> None:
        """Delete the collection and reset the service."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = None
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not reset collection: {e}")


# Convenience function for quick usage
def build_vector_store(
    csv_path: str = "data/homes_preprocessed_data.csv",
    reset: bool = False,
) -> VectorStoreService:
    """
    Build vector store from CSV file.

    Args:
        csv_path: Path to CSV file (relative to project root).
        reset: If True, reset existing collection before building.

    Returns:
        VectorStoreService instance.
    """
    service = VectorStoreService()
    service.build_from_csv(csv_path, reset_collection=reset)
    return service


if __name__ == "__main__":
    # CLI for building vector store
    import argparse

    parser = argparse.ArgumentParser(description="Build ChromaDB vector store from CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/homes_preprocessed_data.csv",
        help="Path to CSV file (relative to project root)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset existing collection before building",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    print(f"Building vector store from: {args.csv}")
    print(f"Reset collection: {args.reset}")
    print(f"Batch size: {args.batch_size}")

    service = VectorStoreService()
    count = service.build_from_csv(
        args.csv,
        batch_size=args.batch_size,
        reset_collection=args.reset,
    )

    print("\nVector store built successfully!")
    print(f"Documents added: {count}")
    print(f"Collection stats: {service.get_collection_stats()}")
