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
    """One evaluation question with its reference answer and expected source pages.

    Attributes:
        id: Short identifier such as "q16", used in printed output and results.
        question: The question as it is asked to the bot.
        answer: Reference answer the judge compares against. For unanswerable
            questions it describes the expected refusal or correction.
        type: "single_fact", "multi_hop" or "unanswerable". Answer scores are
            also reported per type.
        sources: Pages needed to answer, as a list of requirements. Each inner
            list is one requirement and is satisfied by any one of its titles.
            Example: [["2023 UEFA Champions League final"], ["Pep Guardiola"]]
            needs both pages. Titles must exactly match canonical Wikipedia
            titles (as stored in `Chunk.title`). Empty for unanswerable
            questions, which are then left out of retrieval metrics.
    """

    id: str
    question: str
    answer: str
    type: str                     # single_fact | multi_hop | unanswerable
    # Each inner list is one piece of information the question needs, given as the
    # pages where it can be found (any one of them is enough). A multi-hop question
    # has several inner lists. Unanswerable questions have none.
    sources: list[list[str]] = field(default_factory=list)


def load_questions(path: Path) -> list[Question]:
    """Load the evaluation set from a JSON Lines file.

    Each non-blank line is one JSON object whose keys match the fields of
    `Question` (`sources` may be omitted). A line with a missing required key
    or an unknown key raises `TypeError`; invalid JSON raises
    `json.JSONDecodeError`.

    Args:
        path: The questions file, normally `eval/questions.jsonl`.

    Returns:
        The questions in file order.
    """
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [Question(**json.loads(ln)) for ln in lines]


# ---------- retrieval metrics ----------

def retrieval_hit(q: Question, retrieved: list[RetrievedChunk]) -> bool:
    """Did the retrieved chunks cover every page the question needs?

    True only if every requirement in `q.sources` is met by at least one
    retrieved chunk whose title is one of that requirement's titles. Matching
    is by exact title; rank doesn't matter, only presence in the top k.
    For a multi-hop question, finding just one of the needed pages is a miss.

    Only meaningful when `q.sources` is non-empty: with no requirements it
    returns True, which is why `run_eval` skips this for unanswerable questions.

    Args:
        q: The question, with its expected `sources`.
        retrieved: Chunks returned by `FootballRAG.retrieve`.

    Returns:
        True for a hit, False for a miss.
    """
    titles = {c.title for c in retrieved}
    return all(any(t in titles for t in options) for options in q.sources)


def reciprocal_rank(q: Question, retrieved: list[RetrievedChunk]) -> float:
    """Score how high the first relevant chunk was ranked: 1 / its rank.

    A chunk is relevant if its title appears in any of the question's
    requirements. First result relevant -> 1.0, second -> 0.5, third -> 0.33,
    none -> 0.0. Averaged over questions this gives MRR.

    Unlike `retrieval_hit`, this only looks for the FIRST relevant page, so a
    multi-hop question can score 1.0 here while still being a miss.

    Args:
        q: The question, with its expected `sources`.
        retrieved: Chunks in ranked order, most similar first.

    Returns:
        The reciprocal rank, between 0.0 and 1.0.
    """
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
    """Grade the bot's answer against the reference answer with a Claude model ("LLM-as-judge").

    Sends `JUDGE_PROMPT` (question, reference answer, bot answer) to `model`
    with a 200-token cap and parses the first {...} JSON object in the reply.
    The judge sees only the reference answer, not the retrieved sources, so it
    checks correctness, not whether the citations are accurate.

    If the reply can't be parsed, or the verdict isn't one of the three allowed
    values, the answer is counted as "incorrect" and the raw reply (first 200
    characters) is put in "reason", so bad judge output lowers the score
    visibly instead of crashing the run.

    Args:
        client: Anthropic client.
        model: Judge model id (`Settings.judge_model`).
        q: The question, including its reference `answer`.
        answer: The bot's answer text.

    Returns:
        A dict with "verdict" ("correct" | "partial" | "incorrect") and
        "reason" (one sentence). Any extra keys the judge returned are kept.

    Raises:
        anthropic.APIError: The API call itself failed.
    """
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
    """Run the evaluation on every question, print progress, save and summarise the results.

    For each question:
      1. Retrieve the top-k chunks and record their titles.
      2. If the question has expected sources, record `hit` and `rr`
         (reciprocal rank).
      3. Unless `retrieval_only`, also call `rag.ask` (timed as `latency_s`),
         record the answer and token counts, and add the judge's `verdict`
         and `reason`. `ask` retrieves again internally, so each question is
         embedded twice; the results are the same both times.
      4. Print one line: the verdict, or hit/miss in retrieval-only mode, or
         "-" for unanswerable questions in retrieval-only mode.

    Everything is written to `<results_dir>/eval_<YYYYmmdd_HHMMSS>.json` as
    {"summary": ..., "rows": [...]} so runs can be compared later. The saved
    "settings" record the answer model, embedding model, chunk size, chunk
    overlap and top_k (not the judge model).

    Args:
        rag: The pipeline to test. Needs a client unless `retrieval_only`.
        questions: Questions to run, e.g. from `load_questions`.
        judge_model: Model used by `judge`.
        retrieval_only: If True, skip Claude entirely (free, no API key needed).
        results_dir: Directory for the results file; created if missing.

    Returns:
        The summary from `summarize`, plus "settings" and "saved_to" (the
        path of the results file).
    """
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
    """Turn per-question result rows into the headline numbers.

    Always includes "questions" (number of rows). Then, only if relevant rows exist:

      * Retrieval (rows that have a "hit", i.e. questions with expected sources):
        "retrieval_hit@k" = share of hits, "retrieval_mrr" = mean reciprocal rank.
      * Answers (rows that have a "verdict"): "answer_score" = average points
        with correct = 1, partial = 0.5, incorrect = 0; "answer_score_by_type"
        = the same score per question type; "avg_latency_s" = mean time per
        `ask` call; "avg_input_tokens" = mean prompt size sent to Claude.

    All values are rounded (3 decimals for scores, 2 for latency, whole
    tokens). Retrieval-only runs therefore contain no answer keys.

    Args:
        rows: Rows built by `run_eval`.

    Returns:
        The summary dict.
    """
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
