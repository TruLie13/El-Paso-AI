"""Unit checks for step 5 answer insufficiency (no LLM / no Chroma)."""

from langchain_core.documents import Document

from municipal_code_assistant import MunicipalCodeAssistant


def _doc(section: str) -> Document:
    return Document(
        page_content=f"{section} body",
        metadata={"section": section, "title": "Title"},
    )


def main() -> None:
    asst = MunicipalCodeAssistant.__new__(MunicipalCodeAssistant)

    context = [_doc("20.16.030"), _doc("20.02.852")]

    ok = (
        "Residential fences can be up to 6 feet. "
        "Under section 20.16.030 the height limit applies."
    )
    assert asst.answer_insufficiency_reason(ok, context) is None

    short_ok = "Security fences are defined in 20.02.852."
    assert asst.answer_insufficiency_reason(short_ok, context) is None

    bad_cite = "The rule is in section 99.99.999."
    assert asst.answer_insufficiency_reason(bad_cite, context) == "cite_not_in_context"

    # Hedge with no usable cite → weak backup repair.
    hedge = (
        "I would need to see additional information in other sections "
        "to answer that fully."
    )
    assert asst.answer_insufficiency_reason(hedge, context) == "hedge"

    # Soft answer with no cite → do not repair (missing_citation dropped).
    no_cite = "Yes, you can build that fence in a residential area."
    assert asst.answer_insufficiency_reason(no_cite, context) is None

    # Hedge but already cited an in-context section → keep first pass.
    hedge_cited = (
        "Under section 20.16.030 fences may be eight feet, though I would "
        "need additional information about corner-lot exceptions."
    )
    assert asst.answer_insufficiency_reason(hedge_cited, context) is None

    gap = "The provided sections do not answer this question."
    assert asst.answer_insufficiency_reason(gap, context) is None

    empty = asst.answer_insufficiency_reason("   ", context)
    assert empty == "empty_answer"

    selected = asst._select_context_docs(
        [_doc("1.1.1"), _doc("1.1.1"), _doc("2.2.2"), _doc("3.3.3")]
        + [_doc(f"9.9.{i}") for i in range(10)],
        limit=8,
    )
    assert [d.metadata["section"] for d in selected] == [
        "1.1.1",
        "2.2.2",
        "3.3.3",
        "9.9.0",
        "9.9.1",
        "9.9.2",
        "9.9.3",
        "9.9.4",
    ]

    print("answer repair checks ok")


if __name__ == "__main__":
    main()
