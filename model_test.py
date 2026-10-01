#!/usr/bin/env python3
"""
Sequential Ollama model A/B test (not parallel).

Runs the same question through multiple chat models with identical retrieved
context, and prints latency + answer for each.

Usage:
  python model_test.py
  python model_test.py "Can I park near a fire hydrant?"
  python model_test.py -q "can I shit outside" --models llama3 llama3.2:1b
"""

from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from local_embeddings import LocalEmbeddings
from municipal_code_assistant import CODE_COLLECTION, MunicipalCodeAssistant

DEFAULT_MODELS = ["llama3", "llama3.2:1b"]
DEFAULT_QUESTION = "can I shit outside"


def build_assistant_for_retrieval(db_path: str) -> MunicipalCodeAssistant:
    """Vector store + BM25 only — no chat LLM / SelfQuery during retrieve."""
    load_dotenv()
    asst = MunicipalCodeAssistant(db_path=db_path)
    asst.embeddings = LocalEmbeddings()
    asst.vectorstore = Chroma(
        persist_directory=db_path,
        collection_name=CODE_COLLECTION,
        embedding_function=asst.embeddings,
    )
    asst.use_hybrid = True
    asst._rebuild_bm25_index()
    asst.use_self_query = False
    return asst


def generate_with_model(
    asst: MunicipalCodeAssistant,
    model: str,
    question: str,
    context: str,
    base_url: str,
) -> tuple[str, float]:
    """Return (answer_text, elapsed_seconds) for one model."""
    llm = ChatOllama(model=model, base_url=base_url, temperature=0.1)
    asst.llm = llm
    asst.llm_provider = f"ollama:{model}"
    chain = asst._create_summary_chain()

    t0 = time.perf_counter()
    response = chain.invoke({"question": question, "context": context})
    elapsed = time.perf_counter() - t0

    answer = response.content if hasattr(response, "content") else str(response)
    return answer.strip(), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Ollama chat models sequentially on one question."
    )
    parser.add_argument(
        "question_pos",
        nargs="?",
        default=None,
        help="Question text (optional positional)",
    )
    parser.add_argument("-q", "--question", default=None, help="Question text")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to try in order (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument("--db", default="chroma_db", help="Chroma path")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=8,
        help="Sections to retrieve for shared context (default 8)",
    )
    args = parser.parse_args()

    question = args.question or args.question_pos or DEFAULT_QUESTION

    if not os.path.isdir(args.db):
        print(f"Missing {args.db}/ — run python ingest.py first.")
        return 1

    print("=" * 72)
    print("MODEL A/B TEST (sequential)")
    print("=" * 72)
    print(f"Question: {question}")
    print(f"Models:   {' → '.join(args.models)}")
    print()

    print("Retrieving shared context (hybrid search, once)...")
    t_ret0 = time.perf_counter()
    asst = build_assistant_for_retrieval(args.db)
    docs = asst.smart_search_code(question, k=args.k)
    context = asst._format_retrieved_docs(docs)
    t_ret = time.perf_counter() - t_ret0

    sections = [d.metadata.get("section") for d in docs]
    print(f"Retrieve latency: {t_ret:.2f}s")
    print(f"Top sections:     {sections}")
    print()

    results: list[tuple[str, float, str]] = []
    for i, model in enumerate(args.models, start=1):
        print("-" * 72)
        print(f"[{i}/{len(args.models)}] Generating with {model} ...")
        try:
            answer, elapsed = generate_with_model(
                asst, model, question, context, args.base_url
            )
        except Exception as e:
            print(f"FAILED ({model}): {e}")
            results.append((model, float("nan"), f"ERROR: {e}"))
            print()
            continue

        results.append((model, elapsed, answer))
        print(f"Generate latency: {elapsed:.2f}s")
        print()
        print(answer)
        print()

    print("=" * 72)
    print("LATENCY SUMMARY")
    print("=" * 72)
    print(f"{'model':<22} {'generate_s':>10}  note")
    print(f"{'-':-<22} {'-':->10}  {'-':-<20}")
    print(f"{'(shared retrieve)':<22} {t_ret:>10.2f}  same context for all")
    for model, elapsed, answer in results:
        if elapsed != elapsed:  # NaN
            print(f"{model:<22} {'FAIL':>10}")
        else:
            print(f"{model:<22} {elapsed:>10.2f}")

    ok = [e for _, e, a in results if e == e and not str(a).startswith("ERROR:")]
    if len(ok) >= 2:
        fastest = min(ok)
        slowest = max(ok)
        if fastest > 0:
            print(f"\nSpeedup (slowest/fastest generate): {slowest / fastest:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
