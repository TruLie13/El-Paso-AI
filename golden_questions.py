"""Shared golden questions for retrieval evaluation (steps 1–3)."""

# Expected `metadata.section` after step-1 unitization.
GOLDEN_QUESTIONS = [
    {
        "question": "How tall can a residential fence be?",
        "section": "20.16.030",
        "note": "fence / walls height",
    },
    {
        "question": "Can I park near a fire hydrant?",
        "section": "12.44.020",
        "note": "15 feet — where no signs required",
    },
    {
        "question": "parking within fifteen feet of a fire hydrant",
        "section": "12.44.020",
        "note": "lexical hydrant phrasing",
    },
    {
        "question": "can I shit outside",
        "section": "10.16.090",
        "note": "calls of nature / public indecency (topic expand helps)",
    },
    {
        "question": "public urination or defecation prohibited",
        "section": "10.16.090",
        "note": "closer wording to ordinance",
    },
    {
        "question": "dangerous wild animals wolf tiger prohibited",
        "section": "7.08.090",
        "note": "dangerous wild animals",
    },
    {
        "question": "section 12.12.010 authorized emergency vehicles",
        "section": "12.12.010",
        "note": "explicit section id in query",
    },
    {
        "question": "12.44.020",
        "section": "12.44.020",
        "note": "bare section id lookup",
    },
    {
        "question": "What is a security fence?",
        "section": "20.02.852",
        "note": "definition / security fence",
    },
    {
        "question": "crossing a fire hose with a vehicle",
        "section": "12.80.050",
        "note": "crossing fire hose",
    },
]
