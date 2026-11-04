#!/usr/bin/env python3
"""Step 5 smoke: retrieval still works; ask_question returns context_docs."""

from __future__ import annotations

import argparse
import json
import os
import sys

from golden_questions import GOLDEN_QUESTIONS
from local_embeddings import LocalEmbeddings
from langchain_chroma import Chroma
from municipal_code_assistant import CODE_COLLECTION, MunicipalCodeAssistant

DB_PATH = "chroma_db"


def _retrieval_asst() -> MunicipalCodeAssistant:
    asst = MunicipalCodeAssistant(db_path=DB_PATH)
    asst.embeddings = LocalEmbeddings()
    asst.vectorstore = Chroma(
        persist_directory=DB_PATH,
        collection_name=CODE_COLLECTION,
        embedding_function=asst.embeddings,
    )
    asst.use_hybrid = True
    asst._rebuild_bm25_index()
    return asst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default="docs/step5_after.json")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also call ask_question once (needs Ollama/Gemini)",
    )
    args = parser.parse_args()

    if not os.path.isdir(DB_PATH):
        print(f"Missing {DB_PATH}/ — run ingest.py first", file=sys.stderr)
        return 1

    asst = _retrieval_asst()
    ranks = []
    details = []
    print("=== smart_search golden (step 5 must not regress retrieval) ===")
    for item in GOLDEN_QUESTIONS:
        docs = asst.smart_search_code(item["question"], k=8)
        sections = [d.metadata.get("section") for d in docs]
        want = item["section"]
        rank = sections.index(want) + 1 if want in sections else None
        ranks.append(rank)
        details.append({"question": item["question"], "want": want, "rank": rank, "top": sections})
        print(f"  want={want} rank={rank}")

    n = len(ranks)

    def hit(at: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= at) / n if n else 0.0

    summary = {
        "n": n,
        "hit_at_1": hit(1),
        "hit_at_3": hit(3),
        "hit_at_8": hit(8),
        "missing": sum(1 for r in ranks if r is None),
    }
    print(
        f"hit@1={summary['hit_at_1']:.0%} hit@3={summary['hit_at_3']:.0%} "
        f"hit@8={summary['hit_at_8']:.0%}"
    )

    payload = {"smart_search_hybrid": {"summary": summary, "details": details}}

    if args.with_llm:
        print("\n=== ask_question smoke ===")
        full = MunicipalCodeAssistant(db_path=DB_PATH)
        full.initialize()
        result = full.ask_question("How tall can a residential fence be?")
        payload["ask_smoke"] = {
            "success": result.get("success"),
            "iterations": result.get("iterations"),
            "repaired": result.get("repaired"),
            "repair_reason": result.get("repair_reason"),
            "pool_n": len(result.get("documents") or []),
            "context_n": len(result.get("context_docs") or []),
            "has_context_docs": "context_docs" in result,
            "answer_preview": (result.get("answer") or "")[:240],
            "error": result.get("error"),
        }
        print(
            f"  success={result.get('success')} iterations={result.get('iterations')} "
            f"repaired={result.get('repaired')} reason={result.get('repair_reason')}"
        )
        print(
            f"  pool={payload['ask_smoke']['pool_n']} "
            f"context_docs={payload['ask_smoke']['context_n']}"
        )
        if not result.get("success") or "context_docs" not in result:
            print("ask_question smoke failed", file=sys.stderr)
            return 1

    os.makedirs("docs", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0 if summary["hit_at_8"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
