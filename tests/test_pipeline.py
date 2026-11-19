"""Tests that run offline: no Wikipedia, no embedding model download, no API key.

A tiny bag-of-words "embedder" and a fake Claude client stand in for the real ones,
so we can check the plumbing (chunking, storage, retrieval, metrics) quickly.
"""

import hashlib
import math
from types import SimpleNamespace

import pytest

from football_rag.chunking import chunk_pages, split_fixed
from football_rag.config import Settings
from football_rag.evaluate import Question, judge, reciprocal_rank, retrieval_hit, summarize
from football_rag.ingest import clean_text, slugify
from football_rag.rag import FootballRAG
from football_rag.store import RetrievedChunk, VectorStore

PAGES = [
    {"title": "2014 FIFA World Cup", "url": "https://en.wikipedia.org/wiki/2014_FIFA_World_Cup",
     "text": "The 2014 FIFA World Cup was held in Brazil. Germany won the final against "
             "Argentina 1–0 after extra time. Mario Götze scored the winning goal."},
    {"title": "Jürgen Klopp", "url": "https://en.wikipedia.org/wiki/J%C3%BCrgen_Klopp",
     "text": "Jürgen Klopp is a German manager. He managed Mainz 05 and Borussia Dortmund "
             "before joining Liverpool in 2015, winning the Champions League in 2019."},
    {"title": "Real Madrid CF", "url": "https://en.wikipedia.org/wiki/Real_Madrid_CF",
     "text": "Real Madrid plays its home matches at the Santiago Bernabéu stadium in Madrid."},
]


class BagOfWordsEmbedder:
    """Hashes words into a fixed-size vector. Crude, but similar texts share words."""

    dims = 256
    stopwords = {"the", "a", "an", "in", "of", "who", "which", "what", "does", "did",
                 "at", "to", "is", "was", "he", "its", "and", "before", "after", "won"}

    def _vec(self, text):
        v = [0.0] * self.dims
        for word in text.lower().split():
            word = word.strip(".,?!'\"")
            if word in self.stopwords:
                continue
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeClaude:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


@pytest.fixture
def rag():
    store = VectorStore(f"test_{id(object())}")  # in-memory, unique per test
    r = FootballRAG(Settings(chunk_size=120, chunk_overlap=20, top_k=2),
                    BagOfWordsEmbedder(), store, FakeClaude("Germany won [1]."))
    r.build_index(PAGES)
    return r


# ---------- chunking ----------

def test_split_fixed_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    pieces = split_fixed(text, size=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
    # overlap: the end of one chunk reappears at the start of the next
    assert pieces[0].split()[-1] in pieces[1]
    # nothing lost: every word appears somewhere
    joined = " ".join(pieces)
    assert all(f"word{i}" in joined for i in range(300))


def test_split_fixed_does_not_cut_words():
    pieces = split_fixed("alpha beta gamma delta epsilon zeta eta theta", size=20, overlap=5)
    words = set("alpha beta gamma delta epsilon zeta eta theta".split())
    for p in pieces:
        assert set(p.split()) <= words


def test_split_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_fixed("text", size=10, overlap=10)


def test_chunk_ids_are_unique():
    chunks = chunk_pages(PAGES, size=60, overlap=10)
    assert len({c.id for c in chunks}) == len(chunks)
    assert {c.title for c in chunks} == {p["title"] for p in PAGES}


# ---------- ingest helpers ----------

def test_clean_text_drops_reference_sections():
    raw = ("Intro text.\n\n== History ==\nFounded in 1892.\n\n=== Early years ===\nDetail.\n\n"
           "== References ==\nSome ref\n\n== External links ==\nlink")
    cleaned = clean_text(raw)
    assert "Founded in 1892." in cleaned and "Detail." in cleaned
    assert "Some ref" not in cleaned and "link" not in cleaned


def test_slugify():
    assert slugify("2003–04 Arsenal F.C. season") == "2003_04_arsenal_f_c_season"


# ---------- retrieval and generation ----------

def test_retrieve_finds_right_page(rag):
    top = rag.retrieve("Who scored the winning goal in the World Cup final?")
    assert top[0].title == "2014 FIFA World Cup"
    top = rag.retrieve("Which stadium does Real Madrid play at?")
    assert top[0].title == "Real Madrid CF"


def test_ask_sends_numbered_sources_to_claude(rag):
    ans = rag.ask("Who won the 2014 World Cup?")
    assert ans.text == "Germany won [1]."
    prompt = rag.client.calls[0]["messages"][0]["content"]
    assert "[1] (from" in prompt and "Question: Who won the 2014 World Cup?" in prompt
    assert "ONLY the numbered sources" in rag.client.calls[0]["system"]


# ---------- metrics ----------

def _chunk(title):
    return RetrievedChunk(id=title, title=title, url="", text="", score=1.0)


def test_retrieval_hit_and_rr():
    q = Question(id="x", question="", answer="", type="multi_hop",
                 sources=[["A", "B"], ["C"]])
    assert retrieval_hit(q, [_chunk("B"), _chunk("C")])
    assert not retrieval_hit(q, [_chunk("A"), _chunk("D")])
    assert reciprocal_rank(q, [_chunk("D"), _chunk("C")]) == 0.5
    assert reciprocal_rank(q, [_chunk("D")]) == 0.0


def test_judge_parses_json_and_handles_garbage():
    q = Question(id="x", question="Q", answer="A", type="single_fact")
    good = FakeClaude('Here you go: {"verdict": "partial", "reason": "missed score"}')
    assert judge(good, "m", q, "ans")["verdict"] == "partial"
    bad = FakeClaude("not json at all")
    assert judge(bad, "m", q, "ans")["verdict"] == "incorrect"


def test_summarize():
    rows = [
        {"type": "single_fact", "hit": True, "rr": 1.0, "verdict": "correct",
         "latency_s": 1.0, "input_tokens": 100},
        {"type": "single_fact", "hit": False, "rr": 0.0, "verdict": "partial",
         "latency_s": 3.0, "input_tokens": 300},
        {"type": "unanswerable", "verdict": "incorrect", "latency_s": 2.0, "input_tokens": 200},
    ]
    s = summarize(rows)
    assert s["retrieval_hit@k"] == 0.5
    assert s["answer_score"] == 0.5
    assert s["answer_score_by_type"] == {"single_fact": 0.75, "unanswerable": 0.0}
