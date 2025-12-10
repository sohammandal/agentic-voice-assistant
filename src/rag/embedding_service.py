"""
Embedding Service using Google's EmbeddingGemma-300m model.

A standalone HTTP REST API service that provides text embeddings.
Can be started/stopped programmatically and used by MCP servers or data pipelines.

Usage:
    # Start server directly
    python -m src.rag.embedding_service

    # Or use run_service.py for programmatic control
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load environment variables from .env file
# Look for .env in project root (parent of src)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Set HuggingFace token from environment
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token  # Alternative env var name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "google/embeddinggemma-300m"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "embedding_model")
DEFAULT_PORT = 8100
EMBEDDING_DIM = 768  # EmbeddingGemma outputs 768-dim embeddings


class EmbeddingRequest(BaseModel):
    """Request model for single text embedding."""

    text: str = Field(..., description="Text to embed")
    is_query: bool = Field(
        default=True,
        description="If True, use query prompt. If False, use document prompt.",
    )


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch text embedding."""

    texts: list[str] = Field(..., description="List of texts to embed")
    is_query: bool = Field(
        default=True,
        description="If True, use query prompts. If False, use document prompts.",
    )


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""

    embedding: list[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model name used")
    dimension: int = Field(..., description="Embedding dimension")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embeddings."""

    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model name used")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings returned")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_name: str
    device: str


# Global model holder
class ModelHolder:
    def __init__(self):
        self.model = None
        self.device = None
        self.is_loaded = False

    def load_model(self):
        """Load the embedding model."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return

        try:
            from sentence_transformers import SentenceTransformer

            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            logger.info(f"Loading model {MODEL_NAME} on {self.device}...")
            logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")

            # Create cache directory if it doesn't exist
            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

            # Load model with cache directory
            # EmbeddingGemma doesn't support float16, use float32 or bfloat16
            model_kwargs = {}
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16

            # Get HF token for authentication
            token = os.getenv("HF_TOKEN")
            if token:
                logger.info("Using HF_TOKEN from environment")
            else:
                logger.warning(
                    "HF_TOKEN not found in environment. Model download may fail if not logged in."
                )

            self.model = SentenceTransformer(
                MODEL_NAME,
                cache_folder=MODEL_CACHE_DIR,
                device=self.device,
                model_kwargs=model_kwargs,
                token=token,
            )

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def encode_query(self, text: str) -> list[float]:
        """Encode a query text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        embedding = self.model.encode_query(text, convert_to_numpy=True)
        if embedding.ndim == 2:
            # If it returns 2D array for single input, take first row
            embedding = embedding[0]
        return embedding.tolist()

    def encode_document(self, text: str) -> list[float]:
        """Encode a document text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        embedding = self.model.encode_document(text, convert_to_numpy=True)
        if embedding.ndim == 2:
            # If it returns 2D array for single input, take first row
            embedding = embedding[0]
        return embedding.tolist()

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple query texts."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        embeddings = self.model.encode_query(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple document texts."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        embeddings = self.model.encode_document(texts, convert_to_numpy=True)
        return embeddings.tolist()


# Global model instance
model_holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load model on startup."""
    # Startup
    logger.info("Starting embedding service...")
    model_holder.load_model()
    yield
    # Shutdown
    logger.info("Shutting down embedding service...")


# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="REST API for text embeddings using EmbeddingGemma-300m",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_holder.is_loaded else "unhealthy",
        model_loaded=model_holder.is_loaded,
        model_name=MODEL_NAME,
        device=model_holder.device or "not_initialized",
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(request: EmbeddingRequest):
    """Generate embedding for a single text."""
    if not model_holder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if request.is_query:
            embedding = model_holder.encode_query(request.text)
        else:
            embedding = model_holder.encode_document(request.text)

        return EmbeddingResponse(
            embedding=embedding, model=MODEL_NAME, dimension=len(embedding)
        )
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_batch(request: BatchEmbeddingRequest):
    """Generate embeddings for multiple texts."""
    if not model_holder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="Empty texts list")

    try:
        if request.is_query:
            embeddings = model_holder.encode_queries(request.texts)
        else:
            embeddings = model_holder.encode_documents(request.texts)

        return BatchEmbeddingResponse(
            embeddings=embeddings,
            model=MODEL_NAME,
            dimension=len(embeddings[0]) if embeddings else EMBEDDING_DIM,
            count=len(embeddings),
        )
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_available_port(start_port: int = DEFAULT_PORT, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    import socket

    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue

    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts}"
    )


def run_server(host: str = "127.0.0.1", port: Optional[int] = None):
    """Run the embedding service server."""
    import uvicorn

    if port is None:
        port = get_available_port()

    logger.info(f"Starting embedding service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the embedding service")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind to"
    )

    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
