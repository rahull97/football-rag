"""Step 3 of RAG: turn text into vectors ("embeddings").

An embedding model maps text to a list of numbers so that texts with similar
meaning end up close together. "Who coached Liverpool in 2005?" lands near a
chunk about Rafael Benítez even though they share few exact words.

We use BAAI/bge-small-en-v1.5: free, runs on a laptop CPU, and good for its
size. The model downloads automatically (~130 MB) the first time.
"""

from typing import Protocol


class Embedder(Protocol):
    """The interface the rest of the code expects from an embedding model.

    Any object with these two methods can be passed to `FootballRAG`. The tests
    use a tiny fake embedder so they run offline without downloading a model;
    a different provider (e.g. an embeddings API) could be swapped in the same way.

    Documents and queries have separate methods because some models embed
    them differently (see `SentenceTransformerEmbedder.embed_query`).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunk texts for storage. Returns one vector per text, in order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one search question. Returns a single vector."""
        ...


class SentenceTransformerEmbedder:
    """`Embedder` backed by a local sentence-transformers model (default: BGE small).

    Runs on your machine (CPU or Apple Silicon); no API key needed. All vectors
    are normalised to length 1, so cosine similarity equals the dot product,
    which matches the cosine distance the vector store uses.
    """

    # BGE models were trained with this instruction in front of search queries
    # (but not in front of documents). Leaving it out makes retrieval worse.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        """Load the model, downloading it into the Hugging Face cache on first use.

        Loading takes a few seconds, so create one instance and reuse it.
        If the model name contains "bge" (case-insensitive), the BGE query
        prefix is turned on for `embed_query`; for any other model no prefix
        is added.

        Args:
            model_name: Hugging Face model id, e.g. "BAAI/bge-small-en-v1.5"
                (`Settings.embedding_model`).
        """
        from sentence_transformers import SentenceTransformer  # slow import, so done lazily

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts for storing in the index.

        Texts are encoded as-is (no prefix), 32 at a time, and normalised.
        A progress bar is shown when there are more than 100 texts.

        Args:
            texts: The chunk texts.

        Returns:
            One normalised vector (list of floats) per text, in the same order.
        """
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=len(texts) > 100)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for searching the index.

        Adds `query_prefix` (the BGE search instruction, or "" for non-BGE
        models) in front of the question before encoding. This asymmetry is
        intentional: BGE was trained with prefixed queries and unprefixed
        documents.

        Args:
            text: The question.

        Returns:
            One normalised vector.
        """
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()
