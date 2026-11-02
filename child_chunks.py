"""
Conditional child chunks for step 4.

Called from ingest only. Search grain is a child when the parent section does
not fit the live embedder window; the parent section id stays the citation unit.
Full parent text is stored beside Chroma because metadata is the wrong place
for the long body.
"""

from __future__ import annotations

import json
import os
import re

from langchain_core.documents import Document

from local_embeddings import LocalEmbeddings

PARENT_STORE_FILENAME = "parent_units.jsonl"
OVERLAP_TOKENS = 32

_CLAUSE_RE = re.compile(r"(?=\([A-Za-z0-9]{1,4}\)\s)")


def child_chunks_enabled() -> bool:
    """
    Off by default.

    Step 4 measurement: tail-set hit@8 rose, and the section-id pin returned the
    parent, but several golden questions ranked worse (hit@1 80% → 50%).
    The plan keeps chunking off unless CHILD_CHUNKS=1 and a reingest.
    """
    raw = (os.getenv("CHILD_CHUNKS") or "").strip().lower()
    if not raw:
        return False
    return raw in ("1", "true", "yes", "on")


def parent_store_path(db_path: str) -> str:
    return os.path.join(db_path, PARENT_STORE_FILENAME)


def write_parent_store(db_path: str, parents: list[Document]) -> str:
    path = parent_store_path(db_path)
    os.makedirs(db_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for doc in parents:
            metadata = {
                key: value
                for key, value in doc.metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            record = {
                "section": metadata.get("section"),
                "page_content": doc.page_content,
                "metadata": metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_parent_store(db_path: str) -> dict[str, Document]:
    path = parent_store_path(db_path)
    parents: dict[str, Document] = {}
    if not os.path.exists(path):
        return parents
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            section = record.get("section")
            if not section:
                continue
            parents[section] = Document(
                page_content=record.get("page_content") or "",
                metadata=dict(record.get("metadata") or {}),
            )
    return parents


def _prefix_and_body(doc: Document) -> tuple[str, str]:
    section = str(doc.metadata.get("section") or "")
    title = str(doc.metadata.get("title") or "")
    prefix = ""
    if section and title:
        prefix = f"{section} - {title}\n\n"
    elif section:
        prefix = f"{section}\n\n"
    text = doc.page_content or ""
    if prefix and text.startswith(prefix):
        return prefix, text[len(prefix) :]
    return "", text


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return parts if len(parts) > 1 else [text]


def _split_clauses(text: str) -> list[str]:
    parts = [part.strip() for part in _CLAUSE_RE.split(text.strip()) if part.strip()]
    return parts if len(parts) > 1 else [text]


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return parts if len(parts) > 1 else [text]


def _hard_split_words(text: str, fits) -> list[str]:
    words = text.split()
    if len(words) <= 1:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    index = 0
    while index < len(words):
        lo, hi = 1, len(words) - index
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            piece = " ".join(words[index : index + mid])
            if fits(piece):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunks.append(" ".join(words[index : index + best]))
        index += best
    return chunks


def _break_until_fit(text: str, fits, stats: dict, *, depth: int = 0) -> list[str]:
    """Split text until each piece fits. Paragraph, then clause, then sentence, then words."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if fits(cleaned):
        return [cleaned]

    paragraphs = _split_paragraphs(cleaned)
    if len(paragraphs) > 1:
        pieces: list[str] = []
        for part in paragraphs:
            pieces.extend(_break_until_fit(part, fits, stats, depth=depth + 1))
        return pieces

    clauses = _split_clauses(cleaned)
    if len(clauses) > 1:
        stats["clause_pieces"] += len(clauses)
        pieces = []
        for part in clauses:
            pieces.extend(_break_until_fit(part, fits, stats, depth=depth + 1))
        return pieces

    sentences = _split_sentences(cleaned)
    if len(sentences) > 1:
        stats["sentence_pieces"] += len(sentences)
        pieces = []
        for part in sentences:
            pieces.extend(_break_until_fit(part, fits, stats, depth=depth + 1))
        return pieces

    stats["hard_pieces"] += 1
    return _hard_split_words(cleaned, fits)


def _pack_with_overlap(pieces: list[str], fits, embeddings: LocalEmbeddings) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if not current:
            current = piece
            continue
        candidate = f"{current}\n\n{piece}"
        if fits(candidate):
            current = candidate
            continue
        chunks.append(current)
        tail = embeddings.token_tail(current, OVERLAP_TOKENS).strip()
        overlapped = f"{tail}\n\n{piece}" if tail else piece
        if tail and tail not in piece and fits(overlapped):
            current = overlapped
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _as_chunk(doc: Document, text: str, index: int, count: int) -> Document:
    metadata = dict(doc.metadata)
    metadata["chunk_index"] = int(index)
    metadata["chunk_count"] = int(count)
    return Document(page_content=text, metadata=metadata)


def _whole_chunk(doc: Document) -> Document:
    return _as_chunk(doc, doc.page_content, 0, 1)


def expand_units_for_index(
    parents: list[Document],
    embeddings: LocalEmbeddings,
    *,
    enabled: bool | None = None,
) -> tuple[list[Document], list[Document], dict]:
    """
    Return (chroma_docs, parent_docs, stats).

    Parents are unchanged. Chroma docs are children, or the parent itself when
    it fits the window, when it is a table, or when chunking is off.
    """
    if enabled is None:
        enabled = child_chunks_enabled()
    limit = embeddings.max_tokens()
    stats = {
        "enabled": enabled,
        "model_name": embeddings.model_name(),
        "max_tokens": limit,
        "overlap_tokens": OVERLAP_TOKENS,
        "parents": len(parents),
        "stored_whole": 0,
        "split_parents": 0,
        "tables_left_whole": 0,
        "chroma_docs": 0,
        "sections_using_clause": 0,
        "sections_using_sentence": 0,
        "sections_using_hard": 0,
        "clause_pieces": 0,
        "sentence_pieces": 0,
        "hard_pieces": 0,
        "parent_vectors": 0,
    }
    if not enabled:
        chroma_docs = [_whole_chunk(doc) for doc in parents]
        stats["stored_whole"] = len(parents)
        stats["chroma_docs"] = len(chroma_docs)
        return chroma_docs, parents, stats

    chroma_docs: list[Document] = []
    for doc in parents:
        if doc.metadata.get("content_type") == "table":
            chroma_docs.append(_whole_chunk(doc))
            stats["tables_left_whole"] += 1
            stats["stored_whole"] += 1
            continue

        prefix, body = _prefix_and_body(doc)

        def fits(piece: str, prefix: str = prefix) -> bool:
            return embeddings.count_tokens(prefix + piece) <= limit

        if fits(body):
            chroma_docs.append(_whole_chunk(doc))
            stats["stored_whole"] += 1
            continue

        before = (
            stats["clause_pieces"],
            stats["sentence_pieces"],
            stats["hard_pieces"],
        )
        pieces = _break_until_fit(body, fits, stats)
        if stats["clause_pieces"] > before[0]:
            stats["sections_using_clause"] += 1
        if stats["sentence_pieces"] > before[1]:
            stats["sections_using_sentence"] += 1
        if stats["hard_pieces"] > before[2]:
            stats["sections_using_hard"] += 1

        packed = _pack_with_overlap(pieces, fits, embeddings)
        fitting: list[str] = []
        for chunk in packed:
            if fits(chunk):
                fitting.append(chunk)
                continue
            # Overlap can push a packed chunk back over the window.
            stats["hard_pieces"] += 1
            stats["sections_using_hard"] += 1
            fitting.extend(_hard_split_words(chunk, fits))
        if not fitting:
            chroma_docs.append(_whole_chunk(doc))
            stats["stored_whole"] += 1
            continue

        count = len(fitting)
        if count == 1:
            chroma_docs.append(_as_chunk(doc, prefix + fitting[0], 0, 1))
            stats["stored_whole"] += 1
            continue

        stats["split_parents"] += 1
        # Whole-section vector stays in the index so a question the opening
        # already answered can keep that rank. Children add the buried spans.
        # MiniLM still truncates this row to the window at embed time.
        parent_vector = _as_chunk(doc, doc.page_content, -1, count)
        parent_vector.metadata["chunk_role"] = "parent"
        chroma_docs.append(parent_vector)
        stats["parent_vectors"] = stats.get("parent_vectors", 0) + 1
        for index, chunk in enumerate(fitting):
            child = _as_chunk(doc, prefix + chunk, index, count)
            child.metadata["chunk_role"] = "child"
            chroma_docs.append(child)

    stats["chroma_docs"] = len(chroma_docs)
    return chroma_docs, parents, stats
