"""
Vector Database Abstraction Layer
Unified interface for multiple vector database providers
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def upsert(self, documents: List[Dict]) -> int:
        """Insert or update documents. Returns count."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """Delete documents by ID. Returns count."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total document count."""
        pass


class PineconeStore(BaseVectorStore):
    """Pinecone vector store implementation (v3+ API)"""

    def __init__(
        self,
        index_name: str,
        namespace: str = "",
        api_key: str = None
    ):
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("Pinecone not installed. Run: pip install pinecone")

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required")

        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

        # Initialize OpenAI for embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding model"""
        try:
            from openai import OpenAI
            self.openai = OpenAI()
            self.embed_model = "text-embedding-3-small"
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.openai.embeddings.create(
            model=self.embed_model,
            input=text
        )
        return response.data[0].embedding

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = self.openai.embeddings.create(
            model=self.embed_model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def upsert(self, documents: List[Dict]) -> int:
        """
        Insert or update documents.

        Each document should have:
        - id: unique identifier
        - text: content to embed
        - metadata: optional dict of metadata
        """
        if not documents:
            return 0

        texts = [doc["text"] for doc in documents]
        embeddings = self._embed_batch(texts)

        vectors = [
            {
                "id": doc["id"],
                "values": embedding,
                "metadata": {
                    "text": doc["text"][:1000],  # Store truncated text
                    **doc.get("metadata", {})
                }
            }
            for doc, embedding in zip(documents, embeddings)
        ]

        self.index.upsert(vectors=vectors, namespace=self.namespace)
        return len(vectors)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self._embed(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace
        )

        return [
            {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            }
            for match in results.matches
        ]

    def delete(self, ids: List[str]) -> int:
        """Delete documents by ID"""
        self.index.delete(ids=ids, namespace=self.namespace)
        return len(ids)

    def count(self) -> int:
        """Return total document count"""
        stats = self.index.describe_index_stats()
        if self.namespace:
            return stats.namespaces.get(self.namespace, {}).get("vector_count", 0)
        return stats.total_vector_count


class ChromaStore(BaseVectorStore):
    """Chroma vector store implementation (local, no API keys needed)"""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma_db"
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Chroma not installed. Run: pip install chromadb")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, documents: List[Dict]) -> int:
        """Insert or update documents (Chroma handles embeddings)"""
        if not documents:
            return 0

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {
                "id": id,
                "text": doc,
                "metadata": meta,
                "score": 1 - dist  # Convert distance to similarity
            }
            for id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def delete(self, ids: List[str]) -> int:
        """Delete documents by ID"""
        self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        """Return total document count"""
        return self.collection.count()


class VectorStore:
    """Factory for creating vector store instances"""

    PROVIDERS = {
        "pinecone": PineconeStore,
        "chroma": ChromaStore
    }

    def __new__(cls, provider: str, **kwargs) -> BaseVectorStore:
        """
        Create a vector store instance.

        Args:
            provider: "pinecone" or "chroma"
            **kwargs: Provider-specific arguments

        For Pinecone:
            - index_name: Name of the Pinecone index
            - namespace: Optional namespace
            - api_key: Optional API key (defaults to PINECONE_API_KEY env var)

        For Chroma:
            - collection_name: Name of the collection
            - persist_directory: Where to store data (default: ./chroma_db)
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(cls.PROVIDERS.keys())}"
            )

        return cls.PROVIDERS[provider](**kwargs)


def demo():
    """Run a demo with Chroma (no API keys needed)"""
    print("Vector Database Quickstart Demo")
    print("=" * 40)

    # Create store
    store = VectorStore(provider="chroma", collection_name="demo")
    print(f"Created Chroma store. Current count: {store.count()}")

    # Sample documents
    documents = [
        {
            "id": "1",
            "text": "Python is a versatile programming language used for web development, data science, and AI.",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "id": "2",
            "text": "Machine learning is a subset of AI that enables systems to learn from data.",
            "metadata": {"topic": "ai", "subtopic": "ml"}
        },
        {
            "id": "3",
            "text": "Vector databases store high-dimensional embeddings for similarity search.",
            "metadata": {"topic": "databases", "type": "vector"}
        },
        {
            "id": "4",
            "text": "Neural networks are inspired by the human brain and consist of layers of nodes.",
            "metadata": {"topic": "ai", "subtopic": "deep-learning"}
        },
        {
            "id": "5",
            "text": "REST APIs use HTTP methods to create, read, update, and delete resources.",
            "metadata": {"topic": "programming", "type": "api"}
        }
    ]

    # Upsert
    count = store.upsert(documents)
    print(f"Upserted {count} documents. Total: {store.count()}")

    # Search
    queries = [
        "What is artificial intelligence?",
        "How do I build web applications?",
        "database storage systems"
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        results = store.search(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"   {i}. [{result['score']:.3f}] {result['text'][:60]}...")

    # Cleanup
    store.delete(["1", "2", "3", "4", "5"])
    print(f"\nCleaned up. Final count: {store.count()}")


if __name__ == "__main__":
    demo()
