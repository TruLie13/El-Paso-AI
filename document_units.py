"""
PDF-agnostic (gov/corporate) text → document units.

Stage B of the pipeline: chrome cleanup, heading segmentation,
canonicalization, and light structure tagging — before embedding.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document

# --- Chrome / page furniture -------------------------------------------------

_PAGE_MARKER_RE = re.compile(r"(?m)^\s*---\s*page\s+\d+\s*---\s*$", re.I)
_ABOUT_BLANK_LINE_RE = re.compile(r"(?m)^\s*about:blank\b.*$", re.I)
_PAGE_FRACTION_RE = re.compile(r"(?m)^\s*\d{1,4}\s*/\s*\d{1,4}\s*$")
_TIMESTAMP_RE = re.compile(
    r"(?m)^\s*\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)\b.*$",
    re.I,
)
_CODE_OF_ORDINANCES_RE = re.compile(
    r"(?m)^\s*.{0,100}\bCode of Ordinances\s*$", re.I
)
# Inline leftovers after partial line edits
_INLINE_ABOUT_BLANK_RE = re.compile(r"about:blank", re.I)
_INLINE_PAGE_MARK_RE = re.compile(r"---\s*page\s+\d+\s*---", re.I)


def strip_page_chrome(text: str) -> str:
    """Remove recurring PDF/browser chrome without city-specific hardcodes."""
    text = _PAGE_MARKER_RE.sub("", text)
    text = _ABOUT_BLANK_LINE_RE.sub("", text)
    text = _PAGE_FRACTION_RE.sub("", text)
    text = _TIMESTAMP_RE.sub("", text)
    text = _CODE_OF_ORDINANCES_RE.sub("", text)
    text = _INLINE_ABOUT_BLANK_RE.sub("", text)
    text = _INLINE_PAGE_MARK_RE.sub("", text)

    # Drop very common short lines (running headers/footers)
    lines = text.splitlines()
    counts = Counter(ln.strip() for ln in lines if ln.strip())
    # Appear on many pages: high count, short, little sentence structure
    frequent = {
        s
        for s, n in counts.items()
        if n >= 40 and len(s) <= 120 and not _looks_like_heading_line(s)
    }
    if frequent:
        lines = [ln for ln in lines if ln.strip() not in frequent]

    # Collapse excess blank lines
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def _looks_like_heading_line(s: str) -> bool:
    return bool(
        re.match(
            r"^(?:\d{1,2}\.\d{1,2}\.\d{1,3}\b|Section\s+\d|§\s*\d|Article\s+[IVXLC\d])",
            s,
            re.I,
        )
    )


# --- Heading dialects (gov / corporate) --------------------------------------

# Municipal-style: 20.16.030 - Title   OR  20.16.030. Title
_MUNI_HEADER_RE = re.compile(
    r"(?m)^(?P<indent>\s*)"
    r"(?P<section>\d{1,2}\.\d{1,2}\.\d{1,3})\s*"
    r"(?:[-—–.:)]\s*|\s+)"
    r"(?P<title>\S[^\n]{0,200})?\s*$"
)

# Section 4.2 / Section 4.2.1 — Title
_SECTION_WORD_RE = re.compile(
    r"(?m)^(?P<indent>\s*)"
    r"(?P<section>Section\s+\d+(?:\.\d+){0,4})\s*"
    r"(?:[-—–.:)]\s*|\s+)?"
    r"(?P<title>\S[^\n]{0,200})?\s*$",
    re.I,
)

# § 4.2 Title
_SECTION_SYMBOL_RE = re.compile(
    r"(?m)^(?P<indent>\s*)"
    r"(?P<section>§\s*\d+(?:\.\d+){0,4})\s*"
    r"(?:[-—–.:)]\s*|\s+)?"
    r"(?P<title>\S[^\n]{0,200})?\s*$",
)

# Article III — Title
_ARTICLE_RE = re.compile(
    r"(?m)^(?P<indent>\s*)"
    r"(?P<section>Article\s+(?:[IVXLC]+|\d+(?:\.\d+)*))\s*"
    r"(?:[-—–.:)]\s*|\s+)?"
    r"(?P<title>\S[^\n]{0,200})?\s*$",
    re.I,
)

_HEADER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("municipal_decimal", _MUNI_HEADER_RE),
    ("section_word", _SECTION_WORD_RE),
    ("section_symbol", _SECTION_SYMBOL_RE),
    ("article", _ARTICLE_RE),
]


@dataclass(frozen=True)
class HeaderMatch:
    start: int
    end: int  # end of header line
    section: str
    title: str
    dialect: str


def _normalize_section_id(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw.strip())
    # Canonicalize § spacing
    raw = re.sub(r"^§\s*", "§ ", raw)
    if raw.lower().startswith("section "):
        parts = raw.split(None, 1)
        raw = "Section " + parts[1] if len(parts) == 2 else raw
    if raw.lower().startswith("article "):
        parts = raw.split(None, 1)
        raw = "Article " + parts[1] if len(parts) == 2 else raw
    return raw


def find_headers(text: str) -> list[HeaderMatch]:
    """Find line-start headers only (not mid-paragraph citations)."""
    candidates: list[HeaderMatch] = []
    for dialect, pattern in _HEADER_PATTERNS:
        for m in pattern.finditer(text):
            section = _normalize_section_id(m.group("section"))
            title = (m.group("title") or "").strip()
            # Skip TOC-ish ultra-short titles that are only dots/page nums
            if title and re.fullmatch(r"[.\s\d]+", title):
                title = ""
            # Municipal headers should look like real section lines: require
            # a separator-ish title or known title text; bare "12.08.200" mid
            # lists still match with \s+\S — filter Reserved/empty carefully.
            if dialect == "municipal_decimal":
                # Prefer lines that have a title token after the id
                if not title:
                    continue
            candidates.append(
                HeaderMatch(
                    start=m.start(),
                    end=m.end(),
                    section=section,
                    title=title[:200],
                    dialect=dialect,
                )
            )

    if not candidates:
        return []

    # Resolve overlaps: keep earliest start, then longer match / prefer muni id
    candidates.sort(key=lambda h: (h.start, -(h.end - h.start)))
    chosen: list[HeaderMatch] = []
    last_end = -1
    for h in candidates:
        if h.start < last_end:
            continue
        chosen.append(h)
        last_end = h.end
    return chosen


def _is_table_like(body: str) -> bool:
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    short = sum(1 for ln in lines if len(ln) <= 40)
    digitish = sum(
        1 for ln in lines if sum(ch.isdigit() for ch in ln) >= max(1, len(ln) // 4)
    )
    sentences = sum(1 for ln in lines if ln.endswith((".", ";", ":")))
    short_ratio = short / len(lines)
    digit_ratio = digitish / len(lines)
    sentence_ratio = sentences / len(lines)
    return short_ratio >= 0.55 and digit_ratio >= 0.25 and sentence_ratio <= 0.35


def segment_into_units(
    text: str,
    *,
    min_body_chars: int = 100,
) -> list[Document]:
    """
    Clean → header-split → canonicalize by section id → tag structure.

    Returns LangChain Documents with metadata:
      section, title, heading_dialect, content_type
    """
    cleaned = strip_page_chrome(text)
    headers = find_headers(cleaned)
    if not headers:
        return []

    # Collect raw segments
    by_section: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # values: (body, title, dialect)
    for i, h in enumerate(headers):
        body_start = h.end
        body_end = headers[i + 1].start if i + 1 < len(headers) else len(cleaned)
        body = cleaned[body_start:body_end].strip()
        # Drop leading leftover punctuation from header line artifacts
        body = re.sub(r"^[\s\-—–.:)]+", "", body).strip()
        # Chrome can reappear mid-section at page breaks
        body = strip_page_chrome(body).strip()
        by_section[h.section].append((body, h.title, h.dialect))

    documents: list[Document] = []
    for section, variants in by_section.items():
        # Prefer longest body; keep best non-empty title
        body, title, dialect = max(variants, key=lambda t: len(t[0]))
        for b, t, d in variants:
            if not title and t:
                title = t
                dialect = d
        if len(body) < min_body_chars:
            continue
        content_type = "table" if _is_table_like(body) else "prose"
        page_content = body
        if title:
            # Keep title discoverable in both metadata and text for dense/sparse
            page_content = f"{section} - {title}\n\n{body}"
        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "section": section,
                    "title": title,
                    "heading_dialect": dialect,
                    "content_type": content_type,
                },
            )
        )

    documents.sort(key=lambda d: d.metadata["section"])
    return documents


def structural_stats(text: str, docs: list[Document] | None = None) -> dict:
    """Metrics for step-1 verification."""
    cleaned = strip_page_chrome(text)
    headers = find_headers(cleaned)
    docs = docs if docs is not None else segment_into_units(text)

    chrome_hits = 0
    for d in docs:
        if (
            "about:blank" in d.page_content
            or "Code of Ordinances" in d.page_content
            or re.search(r"---\s*page\s+\d+\s*---", d.page_content, re.I)
        ):
            chrome_hits += 1

    dialects = Counter(d.metadata.get("heading_dialect", "") for d in docs)
    types = Counter(d.metadata.get("content_type", "") for d in docs)
    return {
        "header_matches": len(headers),
        "unique_header_ids": len({h.section for h in headers}),
        "indexed_units": len(docs),
        "units_with_chrome": chrome_hits,
        "table_units": types.get("table", 0),
        "prose_units": types.get("prose", 0),
        "dialects": dict(dialects),
        "avg_body_chars": (
            round(sum(len(d.page_content) for d in docs) / len(docs)) if docs else 0
        ),
    }


# --- Legacy splitter (for before/after comparison only) ----------------------

_LEGACY_SPLIT_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{1,3})")


def legacy_split_stats(text: str, min_length: int = 100) -> dict:
    parts = _LEGACY_SPLIT_RE.split(text)
    ids = parts[1::2]
    bodies = parts[2::2]
    kept = [b for b in bodies if len(b.strip()) > min_length]
    unique_kept = {}
    for sid, body in zip(ids, bodies):
        if len(body.strip()) > min_length:
            if sid not in unique_kept or len(body) > len(unique_kept[sid]):
                unique_kept[sid] = body
    chrome = sum(
        1
        for b in kept
        if "about:blank" in b or "Code of Ordinances" in b[:800]
    )
    return {
        "split_points": len(ids),
        "unique_ids_in_stream": len(set(ids)),
        "kept_after_min_length": len(kept),
        "unique_kept_if_deduped": len(unique_kept),
        "kept_with_early_chrome": chrome,
    }
