"""The full pipeline: question -> embed -> search -> Claude -> answer."""

from anthropic import Anthropic

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder
from .generate import Answer, generate_answer
from .store import RetrievedChunk, VectorStore


class FootballRAG:
    """Connects the pipeline steps: chunking, embedding, vector search and Claude.

    This is the main object the CLI and the evaluation use. Its dependencies
    are passed in rather than created inside, so tests can supply a fake
    embedder, an in-memory store and no Claude client.
    """

    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore,
                 client: Anthropic | None = None):
        """Wire the pipeline together. Nothing is computed or downloaded here.

        Args:
            settings: Chunk size/overlap, top_k and the answer model.
            embedder: Used for both indexing and queries. It must be the same
                model the existing index was built with, or search results
                will be meaningless.
            store: Where chunks are stored and searched.
            client: Anthropic client for `ask`. Leave as None for work that
                needs no LLM (indexing, retrieval-only evaluation).
        """
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.client = client

    def build_index(self, pages: list[dict]) -> int:
        """Rebuild the search index from scratch out of the given pages.

        Steps: split pages into chunks (`chunk_pages`, using
        `settings.chunk_size` / `chunk_overlap`), embed every chunk, empty the
        store (`VectorStore.reset`) and add the new chunks.

        Embedding happens before the store is emptied, so if embedding fails
        the old index is left untouched. Run this again after changing pages,
        chunk settings or the embedding model.

        Args:
            pages: Page dicts with "title", "url", "text" (from `load_pages`).

        Returns:
            The number of chunks indexed.
        """
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.reset()
        self.store.add(chunks, embeddings)
        return len(chunks)

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        """Find the chunks most relevant to a question. No Claude call; free.

        Embeds the question with `embed_query` and searches the store.

        Args:
            question: The user's question.
            k: How many chunks to return. If None (or 0), `settings.top_k` is used.

        Returns:
            Up to `k` chunks, most similar first.
        """
        return self.store.search(self.embedder.embed_query(question), k or self.settings.top_k)

    def ask(self, question: str) -> Answer:
        """Answer a question end to end: retrieve `top_k` chunks, then ask Claude.

        Args:
            question: The user's question.

        Returns:
            An `Answer` with Claude's cited answer, the chunks it was given,
            and token usage.

        Raises:
            RuntimeError: The object was created without an Anthropic client.
            anthropic.APIError: The Claude call failed (see `generate_answer`).
        """
        if self.client is None:
            raise RuntimeError("An Anthropic client is needed to generate answers.")
        chunks = self.retrieve(question)
        return generate_answer(self.client, self.settings.answer_model, question, chunks)
