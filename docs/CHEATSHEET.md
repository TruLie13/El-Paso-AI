# Upgrade Cheat Sheet

Easy reference for what we changed, why, and what improved.  
Full plan: [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) · Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## The pipeline in one glance

```text
PDF → OCR text file → clean sections → vectors in Chroma
                                         +
                                      BM25 keywords
                                         ↓
                         hybrid search → (optional) LLM answer
```

| Stage | What it is | Changed in |
| --- | --- | --- |
| A. OCR text | `full_text_ocr.txt` — raw book as text | Left alone (quality was fine) |
| B. Sections | Cut text into real ordinance units | **Step 1** |
| C. Search | Find the right units for a question | **Steps 2–3** |
| D. Answer | LLM explains the retrieved code | **Prompt rewrite** |

---

## Step 1 — Cleaner sections (ingest)

**Problem:** We split on *every* `12.44.020`-style number, including mid-sentence citations. Page headers (`about:blank`, timestamps) got into the DB. ~9.8k noisy Chroma docs.

**What we did:**
- Strip page chrome
- Split only on **real headers** (`12.44.020 - Title`, plus Section/§/Article styles)
- One document per section id (dedupe)
- Tag prose vs table-ish text
- Wipe + rebuild Chroma from the existing OCR file (**no re-OCR**)

**Result:**

| | Before | After |
| --- | --- | --- |
| Chroma docs | ~9,845 | **3,580** |
| Chrome junk in bodies | Common | ~Gone |
| Meaning | Many fake/fragment “sections” | Mostly real ordinance units |

**Files:** `document_units.py`, `ingest.py`, `verify_step1.py`

**Takeaway:** Better *stuff in the database* — the foundation everything else searches.

---

## Step 2 — Smarter ranking (no chat AI)

**Problem:** Search threw away Chroma distances and re-ranked with city-specific bonuses (`if section starts with 20.16…`). Asking for `12.44.020` by number didn’t reliably fetch that section.

**What we did:**
- Keep **similarity distances** and rank by them
- If the question contains a section id → **pin that section first** (metadata lookup)
- Remove chapter-prefix / phrase score hacks
- Expand a small **golden question** set to measure hit rate

**Result** (10 fixed questions, `smart_search`, no LLM):

| Mode | hit@1 | hit@3 | hit@8 |
| --- | --- | --- | --- |
| Plain vector only | 30% | 50% | 70% |
| **Step 2 smart search** | **50%** | **80%** | **100%** |

**Files:** `municipal_code_assistant.py`, `golden_questions.py`, `verify_step2.py`

**Takeaway:** Better *order* of results. Still dense-only — exact words like “fifteen feet” / “fire hydrant” could rank mid-list.

**Note:** Chat LLM is **not** used here.

---

## Step 3 — Hybrid search (vector + keyword)

**Problem:** Embeddings miss exact legal phrasing; keyword/BM25 catches “hydrant”, “fifteen feet”, etc.

**What we did:**
- Build **BM25** over the **same** Chroma sections (on app startup)
- Fuse BM25 + dense lists with **RRF** (reciprocal rank fusion)
- Keep section-id pins from step 2
- Default **on**; turn off with `HYBRID_SEARCH=0`

**Result** (same 10 questions):

| Mode | hit@1 | hit@3 | hit@8 |
| --- | --- | --- | --- |
| Step 2 dense-only | 50% | 80% | 100% |
| **Step 3 hybrid** | **80%** | **100%** | **100%** |

Examples:
- “can I shit outside” → correct section **#1** (was #2)
- Fence height → **#1**
- Fire hydrant wording → **#2** (was #4–#7)

**Files:** `hybrid_retriever.py`, `municipal_code_assistant.py`, `requirements.txt` (`rank_bm25`)

**Takeaway:** This is the real **vector + keyword** hybrid. Restart `main.py` after pull so BM25 rebuilds.

---

## Prompt — How the LLM talks

**Problem:** Answers opened with “YES” / “Based on the provided sections, I can confidently say…” and echoed archaic code language instead of answering the user.

**What we did:** Rewrote the instruction prompt as:

| Piece | Meaning |
| --- | --- |
| **Persona** | Practical El Paso code helper for ordinary residents |
| **Goal** | Answer from retrieved sections; cite the controlling id |
| **Constraints** | No invented law; no boilerplate openers; no full-section dumps; no “In conclusion…” spam |
| **Output format** | Plain answer → section + rule → short quote if needed → penalties only if in context |

**Takeaway:** Retrieval finds the law; the prompt controls *how* it’s explained. Restart `main.py` to load prompt changes.

---

## Quick “what do I run?” 

| Task | Command |
| --- | --- |
| Rebuild DB after unitizer changes | `python ingest.py` (uses OCR cache if present) |
| Check section quality | `python verify_step1.py` |
| Check search quality | `python verify_step2.py --compare-dense --compare-plain` |
| App | `python main.py` |
| Dense-only search | `HYBRID_SEARCH=0 python main.py` |

---

## What we did *not* change (yet)

| Item | Status |
| --- | --- |
| Re-OCR the PDF | Not needed |
| Child chunk / parent section (step 4) | Future |
| Smarter “needs more search” loop (step 5) | Future |
| UI only shows sections sent to the LLM (step 6) | Future |
| Optional “codes only, no AI” mode | Discussed, not built |

---

## One-line memory aid

1. **Step 1** — Fix the *library shelves* (clean sections).  
2. **Step 2** — Fix *how we sort* what we pull (distances + id lookup).  
3. **Step 3** — Fix *keyword blind spots* (BM25 + vectors).  
4. **Prompt** — Fix *how we speak* the answer.
