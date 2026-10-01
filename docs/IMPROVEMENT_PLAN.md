# Improvement Plan

Living backlog for architecture enhancements. Implements in **pipeline order**; later steps assume earlier contracts so nothing fights or silently breaks the app.

Status key: `proposed` → `planned` → `in progress` → `done` | `wontfix`

**Target corpus:** municipal / government / corporate structured docs (codes, policies, handbooks) — not one city’s PDF quirks, not arbitrary flyers.

---

## Holistic order (follow this)

```text
A  PDF → text          (skip unless OCR is bad; El Paso cache is OK)
B  Text → units         STEP 1  — item 0, then wipe+reingest chroma_db
C  Retrieve             STEP 2–4 — P0 → P1 hybrid → P2 chunking
D  Answer + UX          STEP 5–6 — one repair loop, then grounded display
```

| Step | Name | Pipeline | Touches | Reingest? | Blocks |
| --- | --- | --- | --- | --- | --- |
| **1** | Document-unit hygiene | B | `ingest.py` (+ shared preprocessor) | **Yes** (wipe+rebuild) | Everything that measures retrieval |
| **2** | Retrieval P0 | C | `municipal_code_assistant.py` search/rank | No (unless metadata shape changed in step 1 — already rebuilt) | P1 metrics |
| **3** | Retrieval P1 hybrid | C | search + sparse index built at ingest | **Yes** if sparse index is produced at ingest | P2 optional |
| **4** | Retrieval P2 chunking | C | ingest chunking + retriever parent lookup | **Yes** | — |
| **5** | Unified answer repair loop | D | `ask_question` + checks | No | Clean UX wiring |
| **6** | Grounded UI sections | D | assistant return shape + `tui_interface.py` | No | — |

Do **not** start step 3 until step 2 has a golden-set baseline. Prefer finishing step 1 before trusting step 2 numbers.

**Parallelism:** Steps 5–6 can be sketched early, but **implement after** step 2 so repair/UI sit on the same retrieval API. Do **not** implement old “count &lt; 5 SelfQuery” and “citation verify” as two separate loops — step 5 replaces both.

---

## Compatibility contracts (so steps play nice)

These are non-negotiable while we evolve the system:

1. **Chroma replace, not append.** `ingest.py` already `rmtree`s `chroma_db/`. Every ingest step assumes a full rebuild. No migration scripts.
2. **Keep `metadata["section"]` as the stable unit id** (string). SelfQuery, dedupe, citations, and TUI all key off it today. Add fields (`title`, `content_type`, `heading_dialect`) — don’t rename `section` without a coordinated pass.
3. **Keep `LocalEmbeddings` + collection name** (`full_sections_final`) through steps 1–3 unless step 4 explicitly upgrades embeddings (then reingest + smoke test).
4. **One retrieval entry point.** Evolve `smart_search_code` / `batch_search` in place (or a single replacement used everywhere). Don’t leave SelfQuery, BM25, and dense on three divergent code paths with different ranking.
5. **One answer repair loop.** Step 5 owns “should we search again?” Phrase-sniff, SelfQuery-on-count, and citation failure all collapse into that design — no stacked competing triggers.
6. **Assistant → TUI contract is additive first.** Prefer adding `context_docs` (what the LLM saw) while still returning `documents` (pool) until step 6 switches the UI; avoids breaking `main.py` mid-refactor.
7. **LLM-agnostic.** Default Ollama; optional Gemini. No step hardcodes a provider.
8. **Genre-agnostic scoring.** No new city/chapter prefix bonuses. Remove existing ones in step 2 when the cross-encoder (or score-based rank) lands.
9. **Golden set before claiming wins.** Same fixed questions for step 1 smoke → step 2 baseline → step 3 hybrid lift. Otherwise we can’t tell if hybrid helped.
10. **Leave dead alternate ingest alone** (`load_ocr_to_db.py`) until primary path is stable; don’t dual-write two DBs.

---

## Step 1 — Document-unit hygiene (text → sections)

| | |
| --- | --- |
| **Status** | `done` |
| **Where** | `ingest.py` + `document_units.py` |
| **Why first** | Bad units poison dense **and** future BM25. Hybrid cannot fix chrome/TOC fragments. |

**Work**

| ID | Do |
| --- | --- |
| 1a | Strip **generic** page chrome (recurring headers/footers, page stamps, timestamps) via structural/frequency heuristics — not one city’s title string |
| 1b | Split on **headers**, not inline citations. Heading dialect family: municipal `20.16.030 - Title`, `Section 4.2` / `§ 4.2`, `Article III`, `1.2.3` + title, etc. |
| 1c | Canonicalize: one primary doc per heading id; keep richest body; set `metadata.section`, `metadata.title` |
| 1d | Tag table-ish regions (`content_type=table` vs `prose`) generically |

**Play-nice / non-break**

- Still write LangChain `Document`s into wiped `chroma_db/` with `collection_name="full_sections_final"`.
- Runtime keeps working as long as `section` + `page_content` exist.
- Re-OCR **not** required if `full_text_ocr.txt` is good; **re-embed required** after 1a–1d.

**Acceptance**

- Header-like boundaries dominate; inline citation splits rare  
- ~1 indexed unit per heading id  
- Chrome absent from embedded content  
- Fixtures: municipal-decimal + Section/§ style without city hardcodes  
- Smoke: `python ingest.py` + a few `similarity_search` checks  

**Verification (isolate step 1 — same embedder, plain `similarity_search`, no P0/P1)**

Measure **before** wipe (old splitter / old DB) and **after** reingest. Gains must come from cleaner units, not a new ranker.

| Check | How | Significant if |
| --- | --- | --- |
| **1. Structural scorecard** | Compare old vs new unitization on the OCR text (split count, unique ids, % short/junk, chrome in bodies, inline-citation splits) | Near **1 body per id**; chrome ~0; far fewer short/junk and citation splits |
| **2. Golden retrieval** | 15–30 questions with known correct `section`; old DB vs new DB; `similarity_search(q, k=8)` only | Clear top-8 (and ideally top-3) hit-rate lift, or large drop in “never found” — without regressing easy cases |
| **3. Spot checks** | ~10 random units + a few hard ones (tables, long chapters) | Real header + coherent body; no page chrome; tables tagged or not fake-split |
| **4. E2E (secondary)** | Same questions via `ask_question` | Fewer useless 2nd iterations / wrong cites — noisy (LLM); not the only gate |

**Process:** baseline structure (+ retrieval if `chroma_db` exists) → ship step 1 → wipe/reingest → same scripts → compare. Proceed to step 2 only if structure is clearly fixed **and** golden top-8 improves or holds while structure gets much cleaner.

**Out of scope here:** retrieval bonuses, hybrid, answer loop.

---

## Step 2 — Retrieval P0 (measurable dense + re-rank)

| | |
| --- | --- |
| **Status** | `done` |
| **Pipeline** | C |
| **Where** | `batch_search` / `smart_search_code` / `get_docs_by_section` |
| **Depends on** | Step 1 |

**Work**

1. Use `similarity_search_with_score`; keep distances for rank/cutoff  
2. Direct metadata fetch when the query contains a heading/section id  
3. Cross-encoder (or equivalent) re-rank on top-N — **replace** word-overlap + chapter-prefix bonuses  
4. Lock golden set: “correct `section` in top-8 on first pass”

**Play-nice / non-break**

- Same return type: list of Documents (optionally attach score on metadata for debugging).  
- SelfQuery can remain wired but step 5 will change **when** it runs — don’t expand count&lt;5 behavior here.  
- Topic hard-coded query expansions: leave or lightly trim; don’t grow them (P1 sparse replaces that role).

**Acceptance**

- Baseline metrics recorded on golden set  
- Section-id questions resolve via direct fetch  
- Re-rank beats old heuristic on the same set  

---

## Step 3 — Retrieval P1 (hybrid = vector + keyword)

| | |
| --- | --- |
| **Status** | `proposed` |
| **Pipeline** | C |
| **Depends on** | Step 2 baseline |

**Work**

1. BM25 (or sparse) over the **same units** Chroma holds  
2. Fuse sparse + dense (RRF or weighted)  
3. Run the step-2 re-ranker on the fused pool  

**Play-nice / non-break**

- Build/refresh sparse index in **ingest** (or deterministic rebuild from Chroma on startup) so wipe+reingest stays one command.  
- Fusion lives inside the single search entry point — UI/answer loop unchanged.  
- If hybrid regresses, feature-flag back to dense+P0 re-rank without reverting step 1.

**Acceptance**

- Golden-set lift vs step 2  
- Exact-term questions improve without tanking semantic ones  
- Update `ARCHITECTURE.md` to describe real hybrid  

---

## Step 4 — Retrieval P2 (finer grain on good units)

| | |
| --- | --- |
| **Status** | `proposed` |
| **Pipeline** | C |
| **Depends on** | Steps 1 + 3 (hybrid proven or explicitly kept)

**Work**

1. Child chunks for embedding; parent unit for LLM context  
2. Optional stronger embeddings (explicit reingest)  
3. Richer metadata already started in step 1 (`title`, `content_type`)

**Play-nice / non-break**

- Retriever must resolve child → parent before `_format_retrieved_docs` / citations (still cite parent `section`).  
- SelfQuery metadata filters stay on parent ids.  
- Sparse index in step 3 must index the same grain you search (document policy in one place).

**Acceptance**

- Beats step 3 on golden set enough to justify complexity/re-ingest  

---

## Step 5 — Unified answer repair loop (replaces old items 1 + 4)

| | |
| --- | --- |
| **Status** | `proposed` |
| **Pipeline** | D |
| **Where** | `ask_question`, needs-check helpers, SelfQuery invoke site |
| **Depends on** | Step 2+ preferred (repair should re-call the improved search API) |

**Problem today (two competing ideas)**

- SelfQuery when `len(docs) < 5` (quantity)  
- Second search when answer has hedge phrases (style)  
Neither reliably means “missing answer,” and stacking both would double-spend LLM/search.

**Single desired loop**

1. Retrieve via step 2/3 API → format top context → generate  
2. **Early only:** if retrieval is empty / no-signal → one SelfQuery or broad search before generate  
3. **After generate:** repair if **insufficient** — prefer citation/support failure (ids missing, cites not in context, claim unsupported); keep phrase-sniff as weak backup only  
4. At most one repair retrieve + one regenerate unless we later decide otherwise  
5. Return both `documents` (pool) and `context_docs` (what was formatted into the prompt) for step 6  

**Play-nice / non-break**

- Remove count&lt;5 as primary SelfQuery gate when this ships (don’t leave it beside citation repair).  
- SelfQuery becomes one **tool** inside this loop, not a parallel policy.  
- Prompt changes stay provider-agnostic.

**Acceptance**

- Short-but-sufficient contexts skip repair  
- Insufficient answers repair regardless of doc count  
- No double repair from hedge + SelfQuery + citation all firing independently  

---

## Step 6 — Grounded UI (sections the answer used)

| | |
| --- | --- |
| **Status** | `proposed` |
| **Pipeline** | D |
| **Where** | `tui_interface.py` + assistant result dict |
| **Depends on** | Step 5’s `context_docs` (or equivalent) |

**Work**

- Default label/count = sections that grounded the answer (`context_docs`)  
- Optional disclosure for other retrieved candidates  
- Don’t call the full pool “relevant”

**Play-nice / non-break**

- If `context_docs` absent, fall back to old behavior once; then require the field.  
- No retrieval changes here.

**Acceptance**

- UI count matches shown grounding set  
- Shown set ⊆ prompt context  

---

## Explicitly deferred / out of order on purpose

| Idea | Why not now |
| --- | --- |
| Re-OCR El Paso PDF | Character quality OK; stage B is the bottleneck |
| Growing hard-coded topic query maps | Fights genre-agnostic goal; sparse search supersedes |
| Dual DB (`chroma_db_final_structured`) | Confuses primary path |
| Embedding model swap before step 1–3 | Forces reingest without knowing if units/hybrid were the issue |
| Separate SelfQuery count gate **and** citation loop | Conflicting triggers; merged in step 5 |

---

## Done

### Step 1 — Document-unit hygiene (2026-09-30)

**What changed**

- Added `document_units.py` (chrome strip, header dialects, dedupe, table tag)
- `ingest.py` uses it; still wipe+rebuild `chroma_db/` / `full_sections_final`
- Added `verify_step1.py` + metrics in `docs/step1_baseline.json` / `docs/step1_after.json`

**Verification**

| Metric | Before | After |
| --- | --- | --- |
| Split / kept units | 16,350 splits; 8,641 kept (many dupes/chrome) | **3,580** canonical units |
| Chrome-tainted units | 2,075+ early-chrome kept bodies | **~3** (mostly false positives on legal phrase) |
| Golden plain `similarity_search` hit@3 / hit@8 | 0% / 20% | **40% / 40%** (n=5 starter set) |

Structure win is clear; retrieval lift is real but golden IDs/set should be expanded before over-claiming. Ready for step 2 (P0).

### Step 2 — Retrieval P0 (2026-10-01)

**What changed**

- `similarity_search_with_score` + keep `metadata.distance`; rank by ascending distance
- Exact `get_docs_by_section` pin when question contains a section id
- Removed city/chapter-prefix + phrase heuristic `relevance_score`
- Expanded golden set (`golden_questions.py`); `verify_step2.py`
- Cross-encoder deferred — distance + id pin enough for P0 acceptance vs old heuristic/plain dense

**Verification** (`docs/step2_after.json`, n=10)

| Mode | hit@1 | hit@3 | hit@8 |
| --- | --- | --- | --- |
| Plain `similarity_search` | 30% | 50% | 70% |
| **smart_search (P0)** | **50%** | **80%** | **100%** |

Section-id questions rank #1 via metadata fetch. Ready for step 3 (hybrid) when we want lexical lift on hydrant-style wording.
---

## How we use this doc

1. Implement **one step at a time** in the table order.  
2. After each step: smoke the app (`ingest` if needed → `main.py` ask), update status, note what changed.  
3. Sync [ARCHITECTURE.md](ARCHITECTURE.md) when behavior/contracts change.  
4. New ideas go under deferred or as a new step with an explicit dependency — don’t insert a side path that bypasses the single search or single repair loop.
