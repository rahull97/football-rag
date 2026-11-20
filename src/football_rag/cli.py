"""Command line entry point.

    football-rag ingest            download Wikipedia pages listed in data/pages.txt
    football-rag index             chunk + embed them into the vector store
    football-rag ask "question"    ask a question (add --show-sources to see what was retrieved)
    football-rag eval              score retrieval and answers on eval/questions.jsonl
    football-rag eval --retrieval-only   free, quick check of search quality
"""

import argparse
import json
import sys

from .config import settings


def _build_rag(need_llm: bool):
    """Create a `FootballRAG` wired to the real components, using `settings`.

    Opens the persistent Chroma store in `data/chroma/` and loads the
    sentence-transformers embedding model (takes a few seconds; downloads it
    the first time). Heavy libraries are imported inside this function so that
    `football-rag --help` and `ingest` start quickly.

    Args:
        need_llm: If True, also create an Anthropic client. It reads
            ANTHROPIC_API_KEY from the environment (`.env` is loaded by
            `config.py`). A missing or invalid key does not fail here; it
            fails on the first Claude request.

    Returns:
        A ready-to-use `FootballRAG`, with `client=None` when `need_llm` is False.
    """
    from anthropic import Anthropic

    from .embeddings import SentenceTransformerEmbedder
    from .rag import FootballRAG
    from .store import VectorStore

    store = VectorStore(settings.collection, settings.chroma_dir)
    client = Anthropic() if need_llm else None  # reads ANTHROPIC_API_KEY
    return FootballRAG(settings, SentenceTransformerEmbedder(settings.embedding_model), store, client)


def cmd_ingest(_args):
    """`football-rag ingest`: download the pages in data/pages.txt into data/raw/.

    Needs internet access but no API key. See `ingest.ingest` for details.
    """
    from .ingest import ingest

    print(f"Downloading pages listed in {settings.pages_file} ...")
    paths = ingest(settings.pages_file, settings.raw_dir)
    print(f"Done: {len(paths)} pages saved to {settings.raw_dir}")


def cmd_index(_args):
    """`football-rag index`: chunk and embed the saved pages into data/chroma/.

    Replaces any existing index. Exits with a message if `data/raw/` has no
    pages yet. Needs no API key; needs internet only the first time, to
    download the embedding model.
    """
    from .ingest import load_pages

    pages = load_pages(settings.raw_dir)
    if not pages:
        sys.exit("No pages found. Run `football-rag ingest` first.")
    rag = _build_rag(need_llm=False)
    print(f"Chunking and embedding {len(pages)} pages "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}) ...")
    n = rag.build_index(pages)
    print(f"Done: {n} chunks indexed.")


def cmd_ask(args):
    """`football-rag ask "<question>"`: answer one question and print its sources.

    Prints Claude's answer, then each retrieved source as
    "[n] title (similarity x.xx) url", numbered to match the citations in the
    answer, then the token usage. With `--show-sources`, also prints the first
    400 characters of each chunk (newlines flattened) so you can see what
    Claude was given.

    Needs ANTHROPIC_API_KEY and an existing index. If `index` hasn't been run,
    the store is empty, Claude receives no sources and should reply that it
    doesn't know.

    Args:
        args: Parsed arguments with `question` and `show_sources`.
    """
    rag = _build_rag(need_llm=True)
    ans = rag.ask(args.question)
    print(f"\n{ans.text}\n")
    print("Sources:")
    for i, c in enumerate(ans.sources, 1):
        print(f"  [{i}] {c.title}  (similarity {c.score:.2f})  {c.url}")
        if args.show_sources:
            print("      " + c.text[:400].replace("\n", " ") + " ...")
    print(f"\nTokens: {ans.input_tokens} in, {ans.output_tokens} out")


def cmd_eval(args):
    """`football-rag eval`: score the bot on eval/questions.jsonl and save the run.

    With `--retrieval-only`, only search quality is measured (free, no API
    key). Without it, every question is also answered and graded by Claude
    (costs a few cents). `--limit N` runs only the first N questions, handy
    for a quick check. Prints one line per question and then the summary
    JSON. See `evaluate.run_eval`.

    Args:
        args: Parsed arguments with `retrieval_only` and `limit`.
    """
    from .evaluate import load_questions, run_eval

    questions = load_questions(settings.questions_file)
    if args.limit:
        questions = questions[:args.limit]
    rag = _build_rag(need_llm=not args.retrieval_only)
    print(f"Evaluating {len(questions)} questions ...")
    summary = run_eval(rag, questions, settings.judge_model, args.retrieval_only, settings.results_dir)
    print("\n" + json.dumps(summary, indent=2))


def main():
    """Parse the command line and run the matching `cmd_*` function.

    Registered as the `football-rag` console script in pyproject.toml.
    Each subcommand stores its handler in `func`, which is then called with
    the parsed arguments. The module docstring is used as the `--help` text.
    """
    parser = argparse.ArgumentParser(prog="football-rag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="download Wikipedia pages").set_defaults(func=cmd_ingest)
    sub.add_parser("index", help="chunk and embed pages").set_defaults(func=cmd_index)
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--show-sources", action="store_true", help="print retrieved text")
    p_ask.set_defaults(func=cmd_ask)
    p_eval = sub.add_parser("eval", help="run the evaluation")
    p_eval.add_argument("--retrieval-only", action="store_true", help="skip Claude; free")
    p_eval.add_argument("--limit", type=int, help="only the first N questions")
    p_eval.set_defaults(func=cmd_eval)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
