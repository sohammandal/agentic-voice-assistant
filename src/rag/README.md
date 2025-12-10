# Embedding Service

A standalone HTTP REST API service for generating text embeddings using Google's **EmbeddingGemma-300m** model.

## Features

- 🚀 **REST API** - Simple HTTP endpoints for embeddings
- 📦 **Single & Batch** - Embed one text or many at once
- 🔍 **Query vs Document** - Optimized prompts for retrieval tasks
- 💻 **GPU/CPU Auto-detection** - Uses CUDA/MPS if available
- 🔧 **Programmatic Control** - Start/stop from Python scripts

## Installation

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install sentence-transformers fastapi uvicorn torch requests
```

### 2. HuggingFace Authentication

The model requires accepting Google's license agreement:

1. Go to [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m)
2. Click "Agree and access repository"
3. Get an access token from https://huggingface.co/settings/tokens
4. Add the token to your `.env` file in the project root:

```bash
# .env file
HF_TOKEN=hf_your_token_here
```

## Quick Start

### Option 1: Command Line

```bash
# Start the service (from project root)
python -m src.rag.run_service start

# Check status
python -m src.rag.run_service status

# Stop the service
python -m src.rag.run_service stop
```

### Option 2: Run Server Directly

```bash
# From project root
python -m src.rag.embedding_service --host 127.0.0.1 --port 8100
```

### Option 3: From Python

```python
from src.rag.run_service import EmbeddingServiceManager
from src.rag.embedding_client import EmbeddingClient

# Start service (waits for model to load)
manager = EmbeddingServiceManager()
manager.start()

# Get embeddings
client = EmbeddingClient()
embedding = client.embed_query("What is machine learning?")
print(f"Embedding dimension: {len(embedding)}")

# Batch embeddings
docs = ["First document", "Second document"]
embeddings = client.embed_documents(docs)
print(f"Got {len(embeddings)} embeddings")

# Stop when done
manager.stop()
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check & model status |
| `/embed` | POST | Single text embedding |
| `/embed/batch` | POST | Batch text embeddings |

### Example: Single Embedding

```bash
curl -X POST http://127.0.0.1:8100/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "is_query": true}'
```

### Example: Batch Embeddings

```bash
curl -X POST http://127.0.0.1:8100/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Doc 1", "Doc 2"], "is_query": false}'
```

## Testing

### 1. Start the Service

```bash
python -m src.rag.run_service start
```

First run will download the model (~600MB) to `src/rag/embedding_model/`.

### 2. Test with curl (or PowerShell)

**Linux/macOS:**
```bash
curl http://127.0.0.1:8100/health
```

**Windows PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8100/health"
```

### 3. Test with Python

```python
from src.rag.embedding_client import EmbeddingClient

client = EmbeddingClient()

# Check if ready
if client.is_healthy():
    print("Service is ready!")
    
    # Get an embedding
    emb = client.embed_query("test query")
    print(f"Embedding: {len(emb)} dimensions")
else:
    print("Service not ready")
```

### 4. Stop the Service

```bash
python -m src.rag.run_service stop
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Host | `127.0.0.1` | Bind address |
| Port | `8100` | HTTP port |
| Model | `google/embeddinggemma-300m` | HuggingFace model |
| Cache | `src/rag/embedding_model/` | Model download location |

## Troubleshooting

### "Model not loaded" error
- Wait for the model to finish loading (check `/health` endpoint)
- First run downloads ~600MB, may take a few minutes

### "Cannot connect to embedding service"
- Ensure service is running: `python -m src.rag.run_service status`
- Check if port 8100 is available

### HuggingFace authentication error
- Ensure `HF_TOKEN` is set in your `.env` file
- Accept model license at [HuggingFace](https://huggingface.co/google/embeddinggemma-300m)
- Get a token from https://huggingface.co/settings/tokens

### CUDA out of memory
- The model runs in bfloat16 on GPU (~600MB VRAM)
- Falls back to CPU automatically if no GPU

## File Structure

```
src/rag/
├── README.md              # This file
├── __init__.py            # Module exports
├── embedding_service.py   # FastAPI server
├── embedding_client.py    # Python client
├── run_service.py         # Start/stop manager
└── embedding_model/       # Downloaded model (gitignored)
```
