# Vector Database Quickstart

Simple examples for working with vector databases in AI applications.

## Supported Databases

| Database | Type | Best For |
|----------|------|----------|
| Pinecone | Managed | Production, scale |
| Weaviate | Self-hosted/Managed | Hybrid search |
| Chroma | Local | Development, prototyping |
| Qdrant | Self-hosted | Performance |

## Quick Start

```bash
pip install -r requirements.txt
python examples/pinecone_example.py
```

## Examples

### Basic Operations

```python
from vectordb import VectorStore

# Initialize
store = VectorStore(provider="pinecone", index="my-index")

# Add documents
store.upsert([
    {"id": "1", "text": "Hello world", "metadata": {"source": "greeting"}},
    {"id": "2", "text": "AI is transforming industries", "metadata": {"source": "article"}}
])

# Search
results = store.search("artificial intelligence", top_k=5)
```

### With Embeddings

```python
from vectordb import VectorStore
from openai import OpenAI

client = OpenAI()
store = VectorStore(provider="chroma", collection="docs")

# Generate embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text here"
)
embedding = response.data[0].embedding

# Upsert with embedding
store.upsert_vectors([
    {"id": "1", "vector": embedding, "metadata": {"text": "Your text here"}}
])
```

## Use Cases

- **RAG pipelines** - Store document chunks for retrieval
- **Semantic search** - Find similar content
- **Recommendations** - Find similar items/users
- **Deduplication** - Detect near-duplicates

## License

MIT
