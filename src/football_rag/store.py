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
    """A chunk returned by a search, with how closely it matched the question.

    Attributes:
        id: The chunk id, "<article title>#<position>".
        title: Title of the source article.
        url: URL of the source article.
        text: The chunk's text, exactly as stored.
        score: Cosine similarity between the question and the chunk, computed
            as 1 - Chroma's cosine distance. Higher is more similar; 1.0 means
            identical direction. Scores are only meaningful compared with each
            other for the same embedding model.
    """

    id: str
    title: str
    url: str
    text: str
    score: float  # cosine similarity: 1.0 = same direction, 0 = unrelated


class VectorStore:
    """Thin wrapper around one Chroma collection that stores chunks and searches them.

    The collection uses cosine distance. Each stored record has the chunk id,
    the chunk text as the document, the embedding, and metadata
    {"title", "url", "position"}.
    """

    def __init__(self, collection: str, path=None):
        """Open (or create) the collection.

        Args:
            collection: Collection name (`Settings.collection`, "football").
                Chroma requires 3-512 characters from [a-zA-Z0-9._-].
            path: Directory where Chroma persists data (`Settings.chroma_dir`).
                If None, an in-memory store is used and nothing is saved, which
                is what the tests use.
        """
        # path=None gives an in-memory store (used by the tests).
        self.client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self.name = collection
        self.collection = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    def reset(self):
        """Delete every stored chunk by dropping and recreating the collection.

        Called by `FootballRAG.build_index` before adding new chunks, so a
        re-index never mixes chunks from old settings with new ones. This is
        permanent for a persistent store.
        """
        self.client.delete_collection(self.name)
        self.collection = self.client.get_or_create_collection(
            self.name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500):
        """Store chunks with their precomputed embeddings.

        Records are sent to Chroma in batches of `batch_size` to stay under its
        per-request size limit. `embeddings[i]` must belong to `chunks[i]`.

        Ids must be unique. Duplicate ids within one batch raise Chroma's
        `DuplicateIDError`; an id that is already in the collection is silently
        ignored and the old record is kept (Chroma's `add` does not overwrite).
        Call `reset` first when rebuilding the index.

        Args:
            chunks: The chunks to store.
            embeddings: One vector per chunk, in the same order.
            batch_size: Maximum records per Chroma `add` call.
        """
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[i:i + batch_size],
                metadatas=[{"title": c.title, "url": c.url, "position": c.position} for c in batch],
            )

    def search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        """Find the `k` stored chunks most similar to a query vector.

        Args:
            query_embedding: The question's vector, from `Embedder.embed_query`.
                It must come from the same model used to index the chunks.
            k: How many chunks to return.

        Returns:
            Up to `k` chunks, most similar first. Fewer than `k` are returned
            if the collection holds fewer chunks.
        """
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return [
            RetrievedChunk(id=id_, title=meta["title"], url=meta["url"], text=doc,
                           score=1 - dist)
            for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                            res["metadatas"][0], res["distances"][0])
        ]

    def count(self) -> int:
        """Return the number of chunks currently stored (0 if nothing is indexed)."""
        return self.collection.count()
