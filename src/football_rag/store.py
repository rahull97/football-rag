"""Step 4 of RAG: store vectors so we can search them fast.

A vector database keeps every chunk's embedding and, given a query vector,
returns the nearest chunks. We use Chroma, which runs locally and saves to
data/chroma/. Production systems swap in pgvector, OpenSearch, Pinecone and
so on, but the idea is the same.
"""

from dataclasses import dataclass

import chromadb

from .chunking import Chunk


@dataclass
class RetrievedChunk:
    id: str
    title: str
    url: str
    text: str
    score: float  # cosine similarity: 1.0 = same direction, 0 = unrelated


class VectorStore:
    def __init__(self, collection: str, path=None):
        # path=None gives an in-memory store (used by the tests).
        self.client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self.name = collection
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    def reset(self):
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(
            self.name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[i:i + batch_size],
                metadatas=[{"title": c.title, "url": c.url, "position": c.position} for c in batch],
            )

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return [
            RetrievedChunk(id=id_, title=meta["title"], url=meta["url"], text=doc,
                           score=1 - dist)
            for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                            res["metadatas"][0], res["distances"][0])
        ]

    def count(self) -> int:
        return self.collection.count()
