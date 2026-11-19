"""All settings in one place. Values can be overridden in .env or the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
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
