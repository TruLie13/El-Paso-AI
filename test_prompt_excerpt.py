"""Long sections must expose the query-relevant span to the prompt."""

from municipal_code_assistant import MunicipalCodeAssistant


def main() -> None:
    head = "Section 2.1 - CITY ELECTIONS.\n\nA. Regular Elections. " + ("date details. " * 80)
    recall = (
        "C. Recall. Any elected officer of the City may be removed from office.\n"
        "a petition bearing authentic signatures of a number of registered voters equal "
        "to at least twenty percent of the total number of votes cast in the election "
        "of the officer whose recall is sought;"
    )
    text = head + "\n\n" + recall
    assert len(text) > 1200

    q = "how many signatures needed to recall a city rep"
    excerpt = MunicipalCodeAssistant.excerpt_for_query(text, q, limit=1200)
    assert "twenty percent" in excerpt.lower(), excerpt[:200]
    assert "recall" in excerpt.lower(), excerpt[:200]
    # Must not be only the leading elections calendar text.
    assert "Regular Elections" not in excerpt or "twenty percent" in excerpt.lower()

    short = "short section about fences"
    assert MunicipalCodeAssistant.excerpt_for_query(short, q) == short

    print("prompt excerpt checks ok")


if __name__ == "__main__":
    main()
