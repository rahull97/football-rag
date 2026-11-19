"""Step 3 of RAG: turn text into vectors ("embeddings").

An embedding model maps text to a list of numbers so that texts with similar
meaning end up close together. "Who coached Liverpool in 2005?" lands near a
chunk about Rafael Benítez even though they share few exact words.

We use BAAI/bge-small-en-v1.5: free, runs on a laptop CPU, and good for its
size. The model downloads automatically (~130 MB) the first time.
"""

from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    # BGE models were trained with this instruction in front of search queries
    # (but not in front of documents). Leaving it out makes retrieval worse.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # slow import, so done lazily

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=len(texts) > 100)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()
