"""All settings in one place. Values can be overridden in .env or the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Every tunable knob and file path the project uses, in one immutable object.

    The model names and the chunking/retrieval numbers can be overridden with the
    environment variables of the same name in upper case (ANSWER_MODEL, JUDGE_MODEL,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K), usually set in `.env`.
    The paths are fixed relative to the repository root (`ROOT`).

    Note: the defaults are read from the environment once, when this module is
    first imported. Changing `os.environ` afterwards has no effect; construct
    `Settings(chunk_size=...)` explicitly instead (the tests do this).

    Attributes:
        answer_model: Claude model that writes the final answer in `generate.py`.
        judge_model: Cheaper Claude model that grades answers in `evaluate.py`.
        embedding_model: Hugging Face sentence-transformers model used to embed
            both chunks and questions. Changing it requires re-running `index`.
        chunk_size: Target chunk length in characters (~4 characters per token).
        chunk_overlap: Characters shared by consecutive chunks; must be < chunk_size.
        top_k: Number of chunks retrieved and passed to Claude per question.
        pages_file: List of Wikipedia titles to download, one per line.
        raw_dir: Where `ingest` writes one JSON file per downloaded page.
        chroma_dir: Where Chroma persists the vector index on disk.
        collection: Name of the Chroma collection inside `chroma_dir`.
        questions_file: Evaluation set, one JSON object per line.
        results_dir: Where each evaluation run is saved as a timestamped JSON file.
    """

    answer_model: str = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-haiku-5-5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Chunking: sizes are in characters (roughly 4 characters per token).
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # How many chunks to retrieve and hand to Claude for each question.
    top_k: int = int(os.getenv("TOP_K", "5"))

    pages_file: Path = ROOT / "data" / "pages.txt"
    raw_dir: Path = ROOT / "data" / "raw"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection: str = "football"
    questions_file: Path = ROOT / "eval" / "questions.jsonl"
    results_dir: Path = ROOT / "eval" / "results"


settings = Settings()
