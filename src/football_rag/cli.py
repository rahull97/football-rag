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
    from anthropic import Anthropic

    from .embeddings import SentenceTransformerEmbedder
    from .rag import FootballRAG
    from .store import VectorStore

    store = VectorStore(settings.collection, settings.chroma_dir)
    client = Anthropic() if need_llm else None  # reads ANTHROPIC_API_KEY
    return FootballRAG(settings, SentenceTransformerEmbedder(settings.embedding_model), store, client)


def cmd_ingest(_args):
    from .ingest import ingest

    print(f"Downloading pages listed in {settings.pages_file} ...")
    paths = ingest(settings.pages_file, settings.raw_dir)
    print(f"Done: {len(paths)} pages saved to {settings.raw_dir}")


def cmd_index(_args):
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
    from .evaluate import load_questions, run_eval

    questions = load_questions(settings.questions_file)
    if args.limit:
        questions = questions[:args.limit]
    rag = _build_rag(need_llm=not args.retrieval_only)
    print(f"Evaluating {len(questions)} questions ...")
    summary = run_eval(rag, questions, settings.judge_model, args.retrieval_only, settings.results_dir)
    print("\n" + json.dumps(summary, indent=2))


def main():
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
