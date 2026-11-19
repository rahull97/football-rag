"""The full pipeline: question -> embed -> search -> Claude -> answer."""

from anthropic import Anthropic

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder
from .generate import Answer, generate_answer
from .store import RetrievedChunk, VectorStore


class FootballRAG:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore,
                 client: Anthropic | None = None):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.client = client

    def build_index(self, pages: list[dict]) -> int:
        """Chunk and embed every page, replacing whatever was indexed before."""
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.reset()
        self.store.add(chunks, embeddings)
        return len(chunks)

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        return self.store.search(self.embedder.embed_query(question), k or self.settings.top_k)

    def ask(self, question: str) -> Answer:
        if self.client is None:
            raise RuntimeError("An Anthropic client is needed to generate answers.")
        chunks = self.retrieve(question)
        return generate_answer(self.client, self.settings.answer_model, question, chunks)
