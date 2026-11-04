"""Step 6: TUI shows context_docs, not the full retrieve pool."""

from langchain_core.documents import Document

from tui_interface import grounding_docs_from_result


def _doc(section: str) -> Document:
    return Document(page_content=f"{section} body", metadata={"section": section})


def main() -> None:
    pool = [_doc(f"1.1.{i}") for i in range(12)]
    context = pool[:8]

    docs, source = grounding_docs_from_result(
        {"documents": pool, "context_docs": context, "success": True}
    )
    assert source == "context"
    assert len(docs) == 8
    assert [d.metadata["section"] for d in docs] == [f"1.1.{i}" for i in range(8)]

    docs, source = grounding_docs_from_result({"documents": pool, "context_docs": []})
    assert source == "context"
    assert docs == []

    docs, source = grounding_docs_from_result({"documents": pool})
    assert source == "pool"
    assert len(docs) == 12

    docs, source = grounding_docs_from_result({})
    assert source == "pool"
    assert docs == []

    print("grounded ui checks ok")


if __name__ == "__main__":
    main()
