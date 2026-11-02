"""Splitter checks for step 4. Uses a whitespace stand-in for the embedder window."""

from langchain_core.documents import Document

from child_chunks import expand_units_for_index


class StubEmbeddings:
    def __init__(self, limit: int = 24):
        self.limit = limit

    def model_name(self) -> str:
        return "stub"

    def max_tokens(self) -> int:
        return self.limit

    def count_tokens(self, text: str) -> int:
        words = text.split()
        return (len(words) if words else 0) + 2

    def token_tail(self, text: str, n: int) -> str:
        words = text.split()
        return " ".join(words[-n:])


def _doc(section: str, body: str, *, title: str = "Title", content_type: str = "prose") -> Document:
    return Document(
        page_content=f"{section} - {title}\n\n{body}",
        metadata={
            "section": section,
            "title": title,
            "heading_dialect": "municipal_decimal",
            "content_type": content_type,
        },
    )


def main() -> None:
    embeddings = StubEmbeddings(limit=24)

    short = _doc("1.1.1", "A short rule about fences.")
    table = _doc(
        "1.1.2",
        " ".join(f"row{i} 12 13" for i in range(40)),
        content_type="table",
    )
    clauses = _doc(
        "1.1.3",
        "(A) " + " ".join(f"alpha{i}" for i in range(30))
        + " (B) " + " ".join(f"beta{i}" for i in range(30))
        + " (C) " + " ".join(f"gamma{i}" for i in range(12)),
    )
    paragraphs = _doc(
        "1.1.4",
        ("word " * 20).strip() + "\n\n" + ("other " * 20).strip(),
    )

    docs, parents, stats = expand_units_for_index(
        [short, table, clauses, paragraphs],
        embeddings,
        enabled=True,
    )
    assert len(parents) == 4
    by_section: dict[str, list] = {}
    for doc in docs:
        by_section.setdefault(doc.metadata["section"], []).append(doc)

    assert len(by_section["1.1.1"]) == 1
    assert by_section["1.1.1"][0].metadata["chunk_count"] == 1
    assert len(by_section["1.1.2"]) == 1, "tables stay one unit"
    assert len(by_section["1.1.3"]) > 1, "clause block must split"
    assert stats["sections_using_clause"] >= 1
    assert len(by_section["1.1.4"]) > 1

    limit = embeddings.max_tokens()
    for doc in docs:
        assert doc.metadata["section"]
        assert "chunk_index" in doc.metadata
        if doc.metadata["content_type"] == "table":
            continue
        if doc.metadata.get("chunk_role") == "parent":
            continue
        count = embeddings.count_tokens(doc.page_content)
        assert count <= limit, (doc.metadata["section"], count, limit)

    for section in ("1.1.3", "1.1.4"):
        roles = [doc.metadata.get("chunk_role") for doc in by_section[section]]
        assert roles.count("parent") == 1
        assert roles.count("child") >= 1

    off, _, off_stats = expand_units_for_index(
        [short, clauses],
        embeddings,
        enabled=False,
    )
    assert off_stats["split_parents"] == 0
    assert len(off) == 2
    assert all(doc.metadata["chunk_count"] == 1 for doc in off)
    print("child chunk checks ok", {k: stats[k] for k in (
        "split_parents",
        "stored_whole",
        "tables_left_whole",
        "sections_using_clause",
        "chroma_docs",
    )})


if __name__ == "__main__":
    main()
