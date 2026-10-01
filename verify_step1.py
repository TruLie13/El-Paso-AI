#!/usr/bin/env python3
"""Step 1 verification: structural scorecard (+ optional retrieval golden check)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from document_units import legacy_split_stats, segment_into_units, structural_stats

OCR_TEXT_CACHE = "full_text_ocr.txt"
DB_PATH = "chroma_db"
COLLECTION = "full_sections_final"

# Small starter golden set — expand over time. section = expected metadata id.
GOLDEN = [
    {
        "question": "How tall can a residential fence be?",
        "section": "20.16.030",
    },
    {
        "question": "parking near a fire hydrant distance",
        "section": "12.20.120",
    },
    {
        "question": "public urination prohibited",
        "section": "10.12.010",
    },
    {
        "question": "dangerous exotic animals tiger",
        "section": "7.24.020",
    },
    {
        "question": "section 12.12.010 authorized emergency vehicles",
        "section": "12.12.010",
    },
]


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def print_structural(text: str) -> list:
    print("=== Legacy splitter (pre-step-1 behavior) ===")
    legacy = legacy_split_stats(text)
    for k, v in legacy.items():
        print(f"  {k}: {v}")

    print("\n=== New unitization (step 1) ===")
    docs = segment_into_units(text)
    stats = structural_stats(text, docs)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Spot sample
    print("\n=== Sample units ===")
    for doc in docs[:3]:
        meta = doc.metadata
        preview = re.sub(r"\s+", " ", doc.page_content)[:140]
        print(
            f"  {meta.get('section')} [{meta.get('content_type')}] "
            f"title={meta.get('title')!r} :: {preview}..."
        )
    return docs


def evaluate_retrieval(k: int = 8) -> dict:
    from langchain_chroma import Chroma
    from local_embeddings import LocalEmbeddings

    if not os.path.isdir(DB_PATH):
        print(f"No {DB_PATH}/ — skip retrieval check.")
        return {}

    vs = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION,
        embedding_function=LocalEmbeddings(),
    )
    print(f"\n=== Retrieval golden (k={k}), collection count={vs._collection.count()} ===")
    results = []
    hits_at_1 = hits_at_3 = hits_at_8 = 0
    for item in GOLDEN:
        q, want = item["question"], item["section"]
        docs = vs.similarity_search(q, k=k)
        sections = [d.metadata.get("section") for d in docs]
        rank = sections.index(want) + 1 if want in sections else None
        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        if rank is not None and rank <= 8:
            hits_at_8 += 1
        print(f"  Q: {q!r}")
        print(f"     want={want} rank={rank} top={sections[:5]}")
        results.append({"question": q, "want": want, "rank": rank, "top": sections})

    n = len(GOLDEN)
    summary = {
        "n": n,
        "hit_at_1": hits_at_1 / n,
        "hit_at_3": hits_at_3 / n,
        "hit_at_8": hits_at_8 / n,
    }
    print(
        f"\n  hit@1={summary['hit_at_1']:.0%}  "
        f"hit@3={summary['hit_at_3']:.0%}  "
        f"hit@8={summary['hit_at_8']:.0%}"
    )
    return {"summary": summary, "details": results}


def validate_expected_sections(docs: list) -> None:
    """Ensure a few known headers survived unitization."""
    by_id = {d.metadata["section"]: d for d in docs}
    # Soft checks — warn if missing (golden sections may differ by code edition)
    for sid in ("12.12.010", "20.16.030"):
        if sid in by_id:
            print(f"  found unit {sid}: title={by_id[sid].metadata.get('title')!r}")
        else:
            print(f"  WARN: expected section {sid} not in units")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=OCR_TEXT_CACHE)
    parser.add_argument("--retrieval", action="store_true", help="Run golden retrieval on chroma_db")
    parser.add_argument("--json-out", default="", help="Write metrics JSON to path")
    args = parser.parse_args()

    if not os.path.exists(args.text):
        print(f"Missing text cache: {args.text}", file=sys.stderr)
        return 1

    text = load_text(args.text)
    docs = print_structural(text)
    print("\n=== Known section presence ===")
    validate_expected_sections(docs)

    payload = {
        "legacy": legacy_split_stats(text),
        "new": structural_stats(text, docs),
    }
    if args.retrieval:
        payload["retrieval"] = evaluate_retrieval()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
