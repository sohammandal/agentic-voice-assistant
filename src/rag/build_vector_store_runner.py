"""
Runner script to start embedding service (if needed), wait until ready,
and build a Chroma vector store from a CSV file.

Usage:
    python -m src.rag.build_vector_store_runner --csv data/homes_preprocessed_data.csv --reset --batch-size 100 --timeout 600

This script prints progress and exits with code 0 on success, non-zero on failure.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.rag.run_service import EmbeddingServiceManager
from src.rag.vector_store_service import VectorStoreService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Start embedding service and build vector store from CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/homes_preprocessed_data.csv",
        help="CSV path relative to project root",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset existing collection before building"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for embedding"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout seconds to wait for embedding model to load",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Embedding service host"
    )
    parser.add_argument("--port", type=int, default=8100, help="Embedding service port")

    args = parser.parse_args()

    manager = EmbeddingServiceManager(host=args.host, port=args.port)

    # Start service if not running (do not block here; we'll explicitly wait)
    if not manager.is_running():
        logger.info("Embedding service not running. Starting (no-wait)...")
        started = manager.start(wait_for_ready=False)
        if not started:
            logger.error("Failed to start embedding service")
            sys.exit(2)
    else:
        logger.info("Embedding service already running")

    # Wait until model loads (may download on first run)
    logger.info(f"Waiting up to {args.timeout}s for model to load...")
    ready = manager.wait_for_ready(timeout=args.timeout)
    if not ready:
        logger.error("Embedding service did not become ready in time")
        sys.exit(3)

    # Build the vector store
    try:
        logger.info("Embedding service ready. Building vector store...")
        service = VectorStoreService()
        count = service.build_from_csv(
            csv_path=args.csv, batch_size=args.batch_size, reset_collection=args.reset
        )
        stats = service.get_collection_stats()
        logger.info(f"Vector store built successfully. Added {count} documents")
        logger.info(f"Collection stats: {stats}")
        print("BUILD_SUCCESS")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Failed to build vector store: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()
