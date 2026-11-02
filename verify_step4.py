#!/usr/bin/env python3
"""Step 4 checks: tail-set contribution, prompt span, section-id pin.

Build the tail file before any step-4 ingest. Score that same file after.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from langchain_chroma import Chroma

from local_embeddings import LocalEmbeddings
from municipal_code_assistant import CODE_COLLECTION, MunicipalCodeAssistant

DB_PATH = "chroma_db"
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(".")


def _assistant() -> MunicipalCodeAssistant:
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


def _summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)

    def hit(at: int) -> float:
        return sum(1 for rank in ranks if rank is not None and rank <= at) / n if n else 0.0

    return {
        "n": n,
        "hit_at_1": hit(1),
        "hit_at_3": hit(3),
        "hit_at_8": hit(8),
        "missing": sum(1 for rank in ranks if rank is None),
    }


def _candidate_sentence(content: str, tail: str, cut: int) -> str | None:
    sentences = [part.strip() for part in _SENTENCE_RE.split(tail) if part.strip()]
    if not sentences:
        words = tail.split()
        if len(words) >= 12:
            sentences = [" ".join(words[:40])]
    head = content[:cut].lower()
    best = None
    for sentence in sentences:
        words = sentence.split()
        if len(sentence) < 40 or len(words) < 8 or len(sentence) > 400:
            continue
        if sentence.lower() in head:
            continue
        if best is None or len(sentence) > len(best):
            best = sentence
    return best


def build_tail_queries(cap: int = 30) -> dict:
    asst = _assistant()
    embeddings = asst.embeddings
    limit = embeddings.max_tokens()
    raw = asst.vectorstore.get(include=["documents", "metadatas"])
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    candidates = []
    over_window = 0
    for content, meta in zip(documents, metadatas):
        metadata = meta or {}
        if not content or metadata.get("content_type") == "table":
            continue
        if embeddings.count_tokens(content) <= limit:
            continue
        over_window += 1
        cut = embeddings.token_prefix_end(content, limit)
        sentence = _candidate_sentence(content, content[cut:], cut)
        if not sentence:
            continue
        section = metadata.get("section")
        title = metadata.get("title") or ""
        prefix = f"{section} - {title}\n\n" if section and title else ""
        if embeddings.count_tokens(prefix + sentence) > limit:
            continue
        start = content.find(sentence, cut)
        if start < 0:
            continue
        candidates.append(
            {
                "question": sentence,
                "section": section,
                "title": title,
                "token_offset": embeddings.count_tokens(content[:start]),
                "char_offset": start,
            }
        )

    counts: dict[str, int] = {}
    for item in candidates:
        key = _norm(item["question"])
        counts[key] = counts.get(key, 0) + 1
    distinctive = [item for item in candidates if counts[_norm(item["question"])] < 3]
    print(
        f"over_window={over_window} candidates={len(candidates)} "
        f"distinctive={len(distinctive)} scoring..."
    )

    misses = []
    for index, item in enumerate(distinctive, start=1):
        docs = asst.smart_search_code(item["question"], k=8)
        sections = [doc.metadata.get("section") for doc in docs]
        rank = sections.index(item["section"]) + 1 if item["section"] in sections else None
        if rank is None:
            misses.append({**item, "before_rank": None, "before_top": sections})
        if index % 50 == 0 or index == len(distinctive):
            print(f"  scored {index}/{len(distinctive)} misses={len(misses)}")

    misses.sort(key=lambda row: row["section"] or "")
    if len(misses) > cap:
        step = len(misses) / cap
        picked = [misses[int(i * step)] for i in range(cap)]
    else:
        picked = misses
    return {
        "max_tokens": limit,
        "model_name": embeddings.model_name(),
        "over_window": over_window,
        "candidates": len(candidates),
        "distinctive": len(distinctive),
        "misses": len(misses),
        "kept": len(picked),
        "questions": picked,
    }


def score_questions(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    questions = payload["questions"] if isinstance(payload, dict) else payload
    asst = _assistant()
    details = []
    ranks: list[int | None] = []
    for item in questions:
        docs = asst.smart_search_code(item["question"], k=8)
        sections = [doc.metadata.get("section") for doc in docs]
        want = item["section"]
        rank = sections.index(want) + 1 if want in sections else None
        ranks.append(rank)
        details.append(
            {
                "question": item["question"],
                "want": want,
                "rank": rank,
                "top": sections,
                "token_offset": item.get("token_offset"),
            }
        )
        print(f"  want={want} rank={rank} top={sections[:5]}")
    summary = _summarize(ranks)
    print(
        f"hit@1={summary['hit_at_1']:.0%} hit@3={summary['hit_at_3']:.0%} "
        f"hit@8={summary['hit_at_8']:.0%} missing={summary['missing']}"
    )
    return {"summary": summary, "details": details}


def prompt_and_pin_check(tail_path: str, sample: int = 5) -> dict:
    with open(tail_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    questions = (payload["questions"] if isinstance(payload, dict) else payload)[:sample]
    asst = _assistant()
    spans = []
    for item in questions:
        docs = asst.smart_search_code(item["question"], k=8)
        context = asst._format_retrieved_docs(docs)
        spans.append(
            {
                "section": item["section"],
                "sentence_in_context": item["question"] in context,
                "context_excerpt": context[:500],
            }
        )
        print(
            f"  span {item['section']}: "
            f"{'in prompt' if item['question'] in context else 'MISSING from prompt'}"
        )

    pin = asst.get_docs_by_section("12.44.020")
    parent = asst.parents.get("12.44.020")
    pin_ok = False
    pin_note = "no hit"
    if pin and parent is not None:
        pin_ok = pin[0].page_content == parent.page_content and not pin[0].metadata.get(
            "matched_child_text"
        )
        pin_note = (
            f"match_type={pin[0].metadata.get('match_type')} "
            f"same_body={pin[0].page_content == parent.page_content}"
        )
    elif pin and parent is None:
        pin_note = "parent store missing"
    print(f"  pin 12.44.020: {pin_note} ok={pin_ok}")
    return {"spans": spans, "pin_ok": pin_ok, "pin_note": pin_note}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-tail", action="store_true")
    parser.add_argument("--score-tail", action="store_true")
    parser.add_argument("--prompt-check", action="store_true")
    parser.add_argument("--tail-file", default="docs/step4_tail_queries.json")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--cap", type=int, default=30)
    args = parser.parse_args()

    if not os.path.isdir(DB_PATH):
        print(f"Missing {DB_PATH}/ — run ingest.py first", file=sys.stderr)
        return 1

    os.makedirs("docs", exist_ok=True)
    if args.build_tail:
        payload = build_tail_queries(cap=args.cap)
        with open(args.tail_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        before = {
            "summary": _summarize([None] * len(payload["questions"])),
            "details": [
                {
                    "question": item["question"],
                    "want": item["section"],
                    "rank": None,
                    "top": item.get("before_top") or [],
                    "token_offset": item.get("token_offset"),
                }
                for item in payload["questions"]
            ],
        }
        before_path = "docs/step4_tail_before.json"
        with open(before_path, "w", encoding="utf-8") as handle:
            json.dump(before, handle, indent=2)
        print(
            f"Wrote {args.tail_file} kept={payload['kept']} "
            f"of {payload['misses']} misses"
        )
        print(f"Wrote {before_path}")
        return 0 if payload["kept"] else 1

    if args.score_tail:
        result = score_questions(args.tail_file)
        out = args.json_out or "docs/step4_tail_after.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote {out}")
        return 0

    if args.prompt_check:
        result = prompt_and_pin_check(args.tail_file)
        out = args.json_out or "docs/step4_prompt_check.json"
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote {out}")
        failed = [row for row in result["spans"] if not row["sentence_in_context"]]
        return 0 if result["pin_ok"] and not failed else 1

    parser.error("Choose --build-tail, --score-tail, or --prompt-check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
