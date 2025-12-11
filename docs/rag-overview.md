# RAG Module

This module provides a complete RAG (Retrieval-Augmented Generation) solution with:
- **Embedding Service**: HTTP REST API for text embeddings using Google's **EmbeddingGemma-300m** model
- **Vector Store Service**: ChromaDB-based vector database with semantic search and metadata filtering

## Features

### Embedding Service
- **REST API** - Simple HTTP endpoints for embeddings
- **Single & Batch** - Embed one text or many at once
- **Query vs Document** - Optimized prompts for retrieval tasks
- **GPU/CPU Auto-detection** - Uses CUDA/MPS if available
- **Programmatic Control** - Start/stop from Python scripts

### Vector Store Service
- **ChromaDB Integration** - Persistent vector database
- **Semantic Search** - Cosine similarity-based retrieval
- **Metadata Filtering** - Advanced query filters on product attributes
- **CSV Data Loading** - Build vector store from preprocessed product data
- **Batch Processing** - Efficient embedding generation and storage

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

### Step 1: Start the Embedding Service

The embedding service must be running before building the vector store.

```bash
# Start the service (from project root)
python -m src.rag.run_service start

# Check status
python -m src.rag.run_service status
```

### Step 2: Build the Vector Store

```bash
# Build from CSV file
python -m src.rag.vector_store_service --csv data/homes_preprocessed_data.csv --reset
```

### Step 3: Use in Python

```python
from src.rag import VectorStoreService, EmbeddingClient

# Initialize services
client = EmbeddingClient()
vector_store = VectorStoreService()

# Search with semantic similarity
results = vector_store.search("comfortable sofa", top_k=5)

# Search with metadata filters
results = vector_store.search(
    "kitchen appliance",
    top_k=5,
    where={"main_category": "home & kitchen"}
)

# Complex metadata filters
results = vector_store.search(
    "stainless steel blender",
    top_k=10,
    where={
        "$and": [
            {"price": {"$lt": 100}},
            {"brand": "Ninja"}
        ]
    }
)
```

---

## Embedding Service

### Command Line Options

```bash
# Option 1: Using run_service manager
python -m src.rag.run_service start

# Option 2: Direct server launch
python -m src.rag.embedding_service --host 127.0.0.1 --port 8100
```

### Python API

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

---

## Vector Store Service

### Building from CSV

The vector store service reads preprocessed product data and creates a searchable vector database.

**Required CSV columns:**
- `embedding_text`: Text content to embed and search
- `metadata`: Product metadata dictionary (as string)
- `product_id`: Unique identifier for each product

**Metadata structure:**
```python
{
    # Identifiers
    "product_id": "B07XYZ123",
    "product_name": "Product Name",
    
    # Categories
    "main_category": "home & kitchen",
    "category_path": "Home & Kitchen | Kitchen & Dining",
    
    # Numeric metadata
    "price": 49.99,
    "shipping_weight_lbs": 2.5,
    "dim_length": 10.0,
    "dim_width": 8.0,
    "dim_height": 6.0,
    
    # Flags
    "is_amazon_seller": True,
    "has_variants": False,
    
    # Text metadata
    "model_number": "ABC-123",
    "brand": "BrandName",
    "image_url": "https://...",
    "product_url": "https://...",
    
    # Computed metadata
    "features": "Feature text...",
    "about_length": 500,
    "spec_length": 200,
    "tech_details_length": 150
}
```

### Usage Examples

#### Build Vector Store

```python
from src.rag.vector_store_service import VectorStoreService

# Initialize service
service = VectorStoreService()

# Build from CSV (requires embedding service running)
count = service.build_from_csv(
    csv_path="data/homes_preprocessed_data.csv",
    reset_collection=True,  # Delete existing data
    batch_size=100
)
print(f"Added {count} documents")
```

#### Semantic Search

```python
# Basic search
results = service.search("comfortable sofa", top_k=5)

# Access results
for i, (doc_id, doc_text, metadata, distance) in enumerate(
    zip(results["ids"], results["documents"], 
        results["metadatas"], results["distances"])
):
    print(f"{i+1}. {metadata['product_name']}")
    print(f"   Price: ${metadata['price']}")
    print(f"   Distance: {distance:.3f}\n")
```

#### Metadata Filtering

```python
# Price filter
results = service.search(
    "blender",
    top_k=10,
    where={"price": {"$lt": 100}}
)

# Category filter
results = service.search(
    "cookware",
    top_k=10,
    where={"main_category": "home & kitchen"}
)

# Brand filter
results = service.search(
    "kitchen appliance",
    top_k=10,
    where={"brand": "KitchenAid"}
)

# Complex AND filter
results = service.search(
    "stainless steel",
    top_k=10,
    where={
        "$and": [
            {"price": {"$gte": 50, "$lte": 200}},
            {"brand": "Cuisinart"},
            {"is_amazon_seller": True}
        ]
    }
)

# OR filter
results = service.search(
    "coffee maker",
    top_k=10,
    where={
        "$or": [
            {"brand": "Keurig"},
            {"brand": "Ninja"}
        ]
    }
)
```

#### Metadata-Only Search

```python
# Search by metadata without semantic similarity
results = service.search_by_metadata(
    where={"brand": "Sony", "price": {"$lt": 150}},
    top_k=20
)
```

#### Get Specific Document

```python
# Retrieve by product ID
doc = service.get_document("B07XYZ123")
if doc:
    print(doc["metadata"]["product_name"])
    print(doc["document"])
```

#### Collection Management

```python
# Get collection statistics
stats = service.get_collection_stats()
print(f"Collection: {stats['name']}")
print(f"Document count: {stats['count']}")

# Delete specific documents
service.delete_documents(["B07XYZ123", "B08ABC456"])

# Reset entire collection
service.reset()
```

### CLI Usage

```bash
# Build with default settings
python -m src.rag.vector_store_service

# Specify CSV file
python -m src.rag.vector_store_service --csv data/homes_preprocessed_data.csv

# Reset collection before building
python -m src.rag.vector_store_service --csv data/homes_preprocessed_data.csv --reset

# Adjust batch size
python -m src.rag.vector_store_service --csv data/homes_preprocessed_data.csv --batch-size 50
```

### ChromaDB Filter Operators

The vector store supports all ChromaDB metadata filter operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal to | `{"brand": {"$eq": "Sony"}}` or `{"brand": "Sony"}` |
| `$ne` | Not equal to | `{"brand": {"$ne": "Sony"}}` |
| `$gt` | Greater than | `{"price": {"$gt": 50}}` |
| `$gte` | Greater than or equal | `{"price": {"$gte": 50}}` |
| `$lt` | Less than | `{"price": {"$lt": 100}}` |
| `$lte` | Less than or equal | `{"price": {"$lte": 100}}` |
| `$in` | In list | `{"brand": {"$in": ["Sony", "Samsung"]}}` |
| `$nin` | Not in list | `{"brand": {"$nin": ["Unknown"]}}` |
| `$and` | Logical AND | `{"$and": [{"price": {"$lt": 100}}, {"brand": "Sony"}]}` |
| `$or` | Logical OR | `{"$or": [{"brand": "Sony"}, {"brand": "Samsung"}]}` |

---

## API Endpoints (Embedding Service)

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
├── README.md                # This file
├── __init__.py              # Module exports
├── embedding_service.py     # FastAPI server for embeddings
├── embedding_client.py      # Python client for embedding service
├── run_service.py           # Start/stop manager for embedding service
├── vector_store_service.py  # ChromaDB vector store with search
└── embedding_model/         # Downloaded model cache (gitignored)
```

## Complete Workflow Example

```python
from src.rag.run_service import EmbeddingServiceManager
from src.rag import VectorStoreService, EmbeddingClient

# 1. Start embedding service
manager = EmbeddingServiceManager()
manager.start()

# 2. Build vector store from CSV
vector_store = VectorStoreService()
count = vector_store.build_from_csv(
    csv_path="data/homes_preprocessed_data.csv",
    reset_collection=True
)
print(f"Built vector store with {count} products")

# 3. Perform searches
# Semantic search
results = vector_store.search("comfortable sofa", top_k=5)

# Filtered search
results = vector_store.search(
    "kitchen appliance under $100",
    top_k=10,
    where={
        "$and": [
            {"main_category": "home & kitchen"},
            {"price": {"$lt": 100}}
        ]
    }
)

# Display results
for i, metadata in enumerate(results["metadatas"], 1):
    print(f"{i}. {metadata['product_name']} - ${metadata['price']}")

# 4. Cleanup
manager.stop()
```
