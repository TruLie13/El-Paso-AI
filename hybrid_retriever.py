"""
BM25 sparse index + RRF fusion for hybrid retrieval (plan step 3).

Built from the same Chroma units as dense search so wipe+reingest stays consistent
when the assistant rebuilds the index on initialize / from an existing DB.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*", re.I)


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer; keeps dotted section-like tokens."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def rrf_fuse(
    rankings: list[list[str]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion over lists of section ids (best-first).

    score(id) = sum_i 1 / (rrf_k + rank_i)
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, section_id in enumerate(ranking, start=1):
            if not section_id:
                continue
            scores[section_id] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass
class BM25Hit:
    document: Document
    score: float
    rank: int


class BM25Index:
    """In-memory BM25 over section units."""

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._section_to_idx: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def ready(self) -> bool:
        return self._bm25 is not None and len(self._docs) > 0

    def build_from_vectorstore(self, vectorstore) -> int:
        """Load all units from Chroma and build BM25. Returns doc count."""
        raw = vectorstore.get(include=["documents", "metadatas"])
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        docs: list[Document] = []
        corpus: list[list[str]] = []
        section_to_idx: dict[str, int] = {}

        for content, meta in zip(documents, metadatas):
            if not content:
                continue
            metadata = dict(meta or {})
            section = metadata.get("section") or f"anon:{len(docs)}"
            metadata["section"] = section
            doc = Document(page_content=content, metadata=metadata)
            section_to_idx[section] = len(docs)
            docs.append(doc)
            corpus.append(tokenize(content))

        if not docs:
            self._docs = []
            self._section_to_idx = {}
            self._bm25 = None
            return 0

        self._docs = docs
        self._section_to_idx = section_to_idx
        self._bm25 = BM25Okapi(corpus)
        return len(docs)

    def search(self, query: str, k: int = 20) -> list[BM25Hit]:
        if not self.ready or not query.strip():
            return []
        assert self._bm25 is not None
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort descending without numpy dependency
        indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[BM25Hit] = []
        for rank, idx in enumerate(indexed[:k], start=1):
            score = float(scores[idx])
            if score <= 0:
                break
            doc = self._docs[idx]
            meta = dict(doc.metadata)
            meta["bm25_score"] = score
            meta["match_type"] = "bm25"
            hits.append(
                BM25Hit(
                    document=Document(page_content=doc.page_content, metadata=meta),
                    score=score,
                    rank=rank,
                )
            )
        return hits

    def get_by_section(self, section_id: str) -> Document | None:
        idx = self._section_to_idx.get(section_id)
        if idx is None:
            return None
        return self._docs[idx]
