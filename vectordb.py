"""
Vector Database Abstraction Layer
Unified interface for multiple vector database providers
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import OpenAI


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def upsert(self, documents: List[Dict]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        pass


class PineconeStore(BaseVectorStore):
    """Pinecone vector store implementation"""

    def __init__(self, index_name: str, namespace: str = ""):
        import pinecone

        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENV")
        )
        self.index = pinecone.Index(index_name)
        self.namespace = namespace
        self.openai = OpenAI()

    def _embed(self, text: str) -> List[float]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def upsert(self, documents: List[Dict]) -> None:
        vectors = []
        for doc in documents:
            embedding = self._embed(doc["text"])
            vectors.append({
                "id": doc["id"],
                "values": embedding,
                "metadata": doc.get("metadata", {})
            })

        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self._embed(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace
        )

        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            }
            for match in results["matches"]
        ]

    def delete(self, ids: List[str]) -> None:
        self.index.delete(ids=ids, namespace=self.namespace)


class ChromaStore(BaseVectorStore):
    """Chroma vector store implementation (local)"""

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db"):
        import chromadb

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, documents: List[Dict]) -> None:
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        return [
            {
                "id": id,
                "text": doc,
                "metadata": meta,
                "distance": dist
            }
            for id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)


class VectorStore:
    """Factory for creating vector store instances"""

    PROVIDERS = {
        "pinecone": PineconeStore,
        "chroma": ChromaStore
    }

    def __new__(cls, provider: str, **kwargs) -> BaseVectorStore:
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        return cls.PROVIDERS[provider](**kwargs)


if __name__ == "__main__":
    # Example usage with Chroma (no API keys needed)
    store = VectorStore(provider="chroma", collection_name="test")

    # Add documents
    store.upsert([
        {"id": "1", "text": "Python is a programming language", "metadata": {"topic": "programming"}},
        {"id": "2", "text": "Machine learning uses algorithms", "metadata": {"topic": "ai"}},
        {"id": "3", "text": "Vector databases store embeddings", "metadata": {"topic": "databases"}}
    ])

    # Search
    results = store.search("AI and algorithms", top_k=2)
    print("Search results:", results)
