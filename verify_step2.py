#!/usr/bin/env python3
"""Step 2 verification: smart_search_code hit@k on golden questions (no LLM)."""

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


def summarize(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    def hit(at: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= at) / n if n else 0.0

    return {
        "n": n,
        "hit_at_1": hit(1),
        "hit_at_3": hit(3),
        "hit_at_8": hit(min(8, k)),
        "missing": sum(1 for r in ranks if r is None),
    }


def eval_smart(k: int = 8) -> dict:
    asst = MunicipalCodeAssistant(db_path=DB_PATH)
    asst.embeddings = LocalEmbeddings()
    asst.vectorstore = Chroma(
        persist_directory=DB_PATH,
        collection_name=CODE_COLLECTION,
        embedding_function=asst.embeddings,
    )
    # No LLM init — retrieval only

    details = []
    ranks: list[int | None] = []
    print(f"=== smart_search_code golden (k={k}) ===")
    for item in GOLDEN_QUESTIONS:
        q, want = item["question"], item["section"]
        docs = asst.smart_search_code(q, k=k)
        sections = [d.metadata.get("section") for d in docs]
        distances = [d.metadata.get("distance") for d in docs]
        match_types = [d.metadata.get("match_type") for d in docs]
        rank = sections.index(want) + 1 if want in sections else None
        ranks.append(rank)
        print(f"  Q: {q!r}")
        print(f"     want={want} rank={rank} types={match_types[:5]}")
        print(f"     top={sections[:5]} dist={distances[:5]}")
        details.append(
            {
                "question": q,
                "want": want,
                "rank": rank,
                "top": sections,
                "distances": distances,
                "match_types": match_types,
            }
        )

    summary = summarize(ranks, k)
    print(
        f"\n  hit@1={summary['hit_at_1']:.0%}  "
        f"hit@3={summary['hit_at_3']:.0%}  "
        f"hit@8={summary['hit_at_8']:.0%}  "
        f"missing={summary['missing']}"
    )
    return {"summary": summary, "details": details}


def eval_plain_dense(k: int = 8) -> dict:
    """Baseline: single-query similarity_search only (no expansions / id pin)."""
    vs = Chroma(
        persist_directory=DB_PATH,
        collection_name=CODE_COLLECTION,
        embedding_function=LocalEmbeddings(),
    )
    ranks: list[int | None] = []
    print(f"\n=== plain similarity_search golden (k={k}) ===")
    for item in GOLDEN_QUESTIONS:
        q, want = item["question"], item["section"]
        docs = vs.similarity_search(q, k=k)
        sections = [d.metadata.get("section") for d in docs]
        rank = sections.index(want) + 1 if want in sections else None
        ranks.append(rank)
        print(f"  Q: {q!r} want={want} rank={rank} top={sections[:3]}")
    summary = summarize(ranks, k)
    print(
        f"\n  hit@1={summary['hit_at_1']:.0%}  "
        f"hit@3={summary['hit_at_3']:.0%}  "
        f"hit@8={summary['hit_at_8']:.0%}"
    )
    return {"summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--json-out", default="docs/step2_after.json")
    parser.add_argument("--compare-plain", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(DB_PATH):
        print(f"Missing {DB_PATH}/ — run ingest.py first", file=sys.stderr)
        return 1

    payload = {"smart_search": eval_smart(k=args.k)}
    if args.compare_plain:
        payload["plain_dense"] = eval_plain_dense(k=args.k)

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
