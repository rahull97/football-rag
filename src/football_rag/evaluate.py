"""Measure how good the bot is, so every change can be judged by numbers.

Two separate scores, because they fail for different reasons:

1. Retrieval (free, no API calls): did the search bring back chunks from the
   right Wikipedia pages? If not, Claude never sees the answer.
   - hit@k:  share of questions where every needed page shows up in the top k
   - MRR:    mean reciprocal rank; 1.0 if the first result is always from a right page,
             0.5 if it's usually second, and so on

2. Answers (uses Claude): is the final answer actually right? A second, cheaper
   Claude model compares the bot's answer with our reference answer and gives a
   verdict. This "LLM-as-judge" approach is how most teams grade free-text answers.
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anthropic import Anthropic

from .rag import FootballRAG
from .store import RetrievedChunk


@dataclass
class Question:
    id: str
    question: str
    answer: str
    type: str                     # single_fact | multi_hop | unanswerable
    # Each inner list is one piece of information the question needs, given as the
    # pages where it can be found (any one of them is enough). A multi-hop question
    # has several inner lists. Unanswerable questions have none.
    sources: list[list[str]] = field(default_factory=list)


def load_questions(path: Path) -> list[Question]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [Question(**json.loads(ln)) for ln in lines]


# ---------- retrieval metrics ----------

def retrieval_hit(q: Question, retrieved: list[RetrievedChunk]) -> bool:
    titles = {c.title for c in retrieved}
    return all(any(t in titles for t in options) for options in q.sources)


def reciprocal_rank(q: Question, retrieved: list[RetrievedChunk]) -> float:
    relevant = {t for options in q.sources for t in options}
    for rank, chunk in enumerate(retrieved, 1):
        if chunk.title in relevant:
            return 1 / rank
    return 0.0


# ---------- answer judging ----------

JUDGE_PROMPT = """You are grading a question-answering bot about football.

Question: {question}
Reference answer: {reference}
Bot's answer: {answer}

Grade the bot's answer against the reference answer:
- "correct": contains the key facts of the reference and nothing that contradicts it. Extra true detail is fine.
- "partial": gets some key facts but misses or garbles others.
- "incorrect": wrong, contradicts the reference, or says it doesn't know when the reference has an answer.
If the reference says the bot should decline or correct a false premise, then declining or correcting is "correct" and inventing an answer is "incorrect".

Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}}"""


def judge(client: Anthropic, model: str, q: Question, answer: str) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=q.question, reference=q.answer, answer=answer)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        result = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        result = {}
    if result.get("verdict") not in {"correct", "partial", "incorrect"}:
        return {"verdict": "incorrect", "reason": f"Judge output could not be parsed: {text[:200]}"}
    return result


# ---------- the evaluation run ----------

def run_eval(rag: FootballRAG, questions: list[Question], judge_model: str,
             retrieval_only: bool, results_dir: Path) -> dict:
    rows = []
    for q in questions:
        retrieved = rag.retrieve(q.question)
        row = {"id": q.id, "type": q.type, "question": q.question,
               "retrieved_titles": [c.title for c in retrieved]}
        if q.sources:
            row["hit"] = retrieval_hit(q, retrieved)
            row["rr"] = reciprocal_rank(q, retrieved)
        if not retrieval_only:
            start = time.perf_counter()
            ans = rag.ask(q.question)
            row["latency_s"] = round(time.perf_counter() - start, 2)
            row["answer"] = ans.text
            row["input_tokens"], row["output_tokens"] = ans.input_tokens, ans.output_tokens
            row.update(judge(rag.client, judge_model, q, ans.text))
        rows.append(row)
        status = row.get("verdict", "hit" if row.get("hit") else "miss" if "hit" in row else "-")
        print(f"  {q.id:<5} {status:<10} {q.question[:70]}")

    summary = summarize(rows)
    summary["settings"] = {k: str(v) for k, v in asdict(rag.settings).items()
                           if k in {"answer_model", "embedding_model", "chunk_size",
                                    "chunk_overlap", "top_k"}}
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    summary["saved_to"] = str(out)
    return summary


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if "hit" in r]
    summary = {"questions": len(rows)}
    if scored:
        summary["retrieval_hit@k"] = round(sum(r["hit"] for r in scored) / len(scored), 3)
        summary["retrieval_mrr"] = round(sum(r["rr"] for r in scored) / len(scored), 3)
    judged = [r for r in rows if "verdict" in r]
    if judged:
        points = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
        summary["answer_score"] = round(sum(points[r["verdict"]] for r in judged) / len(judged), 3)
        by_type = {}
        for r in judged:
            by_type.setdefault(r["type"], []).append(points[r["verdict"]])
        summary["answer_score_by_type"] = {t: round(sum(v) / len(v), 3) for t, v in by_type.items()}
        summary["avg_latency_s"] = round(sum(r["latency_s"] for r in judged) / len(judged), 2)
        summary["avg_input_tokens"] = round(sum(r["input_tokens"] for r in judged) / len(judged))
    return summary
