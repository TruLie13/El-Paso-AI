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
| **4** | Retrieval P2 chunking | C | ingest, parent store, search collapse, prompt span | **Yes** | — |
| **5** | Unified answer repair loop | D | `ask_question` + checks | No | Clean UX wiring |
| **6** | Grounded UI sections | D | assistant return shape + `tui_interface.py` | No | — |

Do **not** start step 3 until step 2 has a golden-set baseline. Prefer finishing step 1 before trusting step 2 numbers.

**Parallelism:** Steps 5–6 can be sketched early, but **implement after** step 2 so repair/UI sit on the same retrieval API. Do **not** implement old “count &lt; 5 SelfQuery” and “citation verify” as two separate loops — step 5 replaces both.

---

## Compatibility contracts (so steps play nice)

These are non-negotiable while we evolve the system:

1. **Chroma replace, not append.** `ingest.py` already `rmtree`s `chroma_db/`. Every ingest step assumes a full rebuild. No migration scripts.
2. **Keep `metadata["section"]` as the stable unit id** (string). SelfQuery, dedupe, citations, and TUI all key off it today. Add fields (`title`, `content_type`, `heading_dialect`) — don’t rename `section` without a coordinated pass.
3. **Keep `LocalEmbeddings` + collection name** (`full_sections_final`) through steps 1–4. Step 4 reads that embedder’s window; it does not swap models. A later embedding change is its own reingest + smoke test.
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
| **Status** | `done` |
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

## Step 4 — Retrieval P2 (conditional child chunks)

| | |
| --- | --- |
| **Status** | `measured — off` |
| **Pipeline** | C |
| **Where** | `local_embeddings.py`, `ingest.py`, parent store, `smart_search_code` / `BM25Index`, `_format_retrieved_docs` |
| **Depends on** | Steps 1 + 3 (hybrid proven or explicitly kept) |

Search grain and answer grain stay different. The heading unit from step 1 is always the parent: that is what we cite and what `metadata.section` means. A child exists only so a sentence past the embedder window can be found. Short sections are stored whole. This is not an El Paso chunk size; the threshold is whatever embedder ingest is about to use.

**Work**

1. **Window from the live embedder.** `LocalEmbeddings.max_tokens()` and `count_tokens()` delegate to the wrapped function (Chroma `ONNXMiniLM_L6_V2` is 256 today). If the wrapper has no `max_tokens()`, ingest fails unless `EMBED_MAX_TOKENS` is set. Record model name and window in the ingest stats. Swapping models still means a full wipe+reingest; the new run reads the new window. No model upgrade in this step.
   - **Count with truncation off.** That MiniLM tokenizer is built with `enable_truncation(max_length=256)`. A count through the truncated tokenizer reports 256 for every long section, the fit check passes, and ingest stores today’s units unchanged. Measure the real length, then decide.
   - The budget is `max_tokens()` minus the `"section - title"` prefix and minus the overlap. A child that ignores either is truncated again at embed time.
2. **One policy function, called only from ingest,** after `segment_into_units`:
   - Body plus prefix fits in the budget → one vector, same as today.
   - Longer **prose** → split so each child fits. Prefer paragraph breaks. Many sections are one block of `(A)` / `(B)` clauses with no blank line; a paragraph-only split leaves that block over the window and MiniLM truncates it again. If a piece still does not fit, split on the next boundary (clause, then sentence). Sample a few long sections before tuning overlap, and record how many pieces needed the fallback.
   - Overlap is small (~32 tokens) and only covers the cut.
   - `content_type=table` stays one unit. Do not split rows off their headers.
   - Each child keeps parent `metadata.section`, `title`, `heading_dialect`, `content_type`, and adds `chunk_index` / `chunk_count`.
3. **Parent text beside Chroma.** Chroma holds the text that was embedded (the child, or the whole section when it was not split). Full parent bodies go in a store wiped and rewritten in the same ingest as `chroma_db/` (Chroma metadata is the wrong place for the long body). `BM25Index` keeps reading Chroma, so it sees children.
4. **Over-fetch, then collapse, then one parent resolve.**
   - Dense and BM25 both search children. `batch_search` and the BM25 pool are small today (`k` per query, BM25 pool `max(20, k*3)`). After chunking, those slots can fill with several pieces of one long section and the right parent never reaches top 8. Pull a wider child pool, then collapse to unique parent section ids (best child rank wins) before RRF. `rrf_fuse` already scores section ids.
   - **One resolve function** turns a Chroma hit into the parent document. It sits on every path that can leave the assistant with `page_content`: the section-id pin, the fusion fallback, and the SelfQuery docs `ask_question` still appends when fewer than 5 docs come back. `get_docs_by_section` returns every child, and the pin currently keeps the first row Chroma happens to return. `BM25Index.get_by_section` keeps only the last row for a section id. Either bug leaves a fragment in the prompt while the golden section-id questions still rank 1.
   - Copy the winning `chunk_index` (or the matched child text) onto the parent metadata before discarding the child body. The prompt step needs that identity after `page_content` becomes the full section.
5. **Matched span in the prompt.** `_format_retrieved_docs` currently keeps the first 1,000 characters of each section, which drops the sentence step 4 just found. Format parent id and title, then the matched child (it already fits the embedder window).
6. **Rollback is an ingest flag.** `CHILD_CHUNKS=0` at ingest indexes parent documents exactly as step 3 does. Setting it only at query time on an already-chunked database does not restore step 3. A regression does not require reverting step 1 or hybrid; it requires a reingest with the flag off.

**Play-nice / non-break**

- Still wipe+rebuild `chroma_db/` and collection `full_sections_final`. Parent store is part of that same replace, not a second database.
- `metadata.section` stays the parent id. SelfQuery filters stay on parent ids. SelfQuery hits still pass through the same parent resolve as other Chroma reads.
- Dense and sparse index the same grain (children when chunking is on; parents when `CHILD_CHUNKS=0`). Fusion and the answer loop still speak in parent section ids.
- Do not add a second retriever (`ParentDocumentRetriever` or a side index). Chunk policy lives in one ingest function. Parent resolve lives in one function used by every Chroma read on the answer path.
- Genre-agnostic: no city chunk sizes, no fixed character width tuned on this code. A shorter SaaS corpus mostly stays whole because its sections fit the window.

**Verification (freeze before any step 4 ingest)**

Step 4 wipes `chroma_db/`. Capture both baselines first. Compare hybrid `smart_search_code` only. Plain `similarity_search` lists several chunks of one section and makes hit@k look worse for a reason that is not the ranker. No LLM in the rank comparison.

1. **Golden regression set** — the 10 questions in `golden_questions.py` (same set as `docs/step3_after.json`). Historical hybrid result is hit@1 80% / hit@3 100% / hit@8 100%; only the two fire-hydrant questions are rank 2, and the two section-id questions never touch the embedder. Re-run on the live database and keep that file. Do not trust the October numbers if the database has moved.

   ```bash
   python verify_step2.py --json-out docs/step4_before.json
   ```

   After implementation, the same command writes `docs/step4_after.json`. Diff **per-question rank**, not only the percentages.

2. **Tail set (this is the contribution test).** The golden set is already saturated, so a tie there does not show that child chunks helped. Before wiping the database, build `docs/step4_tail_queries.json` from the current units, then freeze it. The after run uses that file unchanged.

   - Start from prose sections longer than the live window (truncation off, same count as ingest). Skip tables.
   - Candidate query: a sentence from past the first `max_tokens()`, not one that also appears inside that prefix.
   - Drop boilerplate that repeats across many sections (“as provided in this chapter” matches everywhere and cannot rise).
   - Score the candidates with today’s `smart_search_code` and **keep the misses**: outside top 8. Cap the file at a few dozen. A distinctive sentence is often already found by BM25 on the full parent; scoring every long section dilutes the lift and can hide a real dense-search gain. The interesting rows are the ones the current hybrid does not already return.
   - Save the query, the parent section id, and the token offset.

   Ranks now → `docs/step4_tail_before.json` (these should be misses by construction). After step 4 → `docs/step4_tail_after.json`.

3. **Prompt span, and the pin path.** Rank does not prove the model saw the sentence. `_format_retrieved_docs` currently keeps the first 1,000 characters. On a handful of tail queries, save the formatted context before and after. After step 4 the tail sentence must appear in that string. Separately, a section-id question (`12.44.020`) must resolve to the parent body. Golden rank stays 1 even when the pin returns the first child fragment, so rank diff will not catch that.

4. **Ingest stats** on the step 4 run: embedder name, `max_tokens()`, parents split vs stored whole, tables left whole, and how many pieces needed the clause or sentence fallback. A fallback count of zero on a corpus of long single blocks means the splitter never saw past the paragraph boundary.

5. **Rollback proof.** Reingest with `CHILD_CHUNKS=0` (ingest time, not a query-time switch) and score the golden file again. Ranks must match `docs/step4_before.json`. That is the path that puts retrieval back without reverting step 1 or hybrid. If the flag does not reproduce the before file, the rollback is not real and step 4 does not ship.

**Keep step 4 only if all of these hold**

- Golden: no question’s rank gets worse than `docs/step4_before.json`. A tie is acceptable. A drop is not.
- Tail: hit@8 (and ideally hit@3) rises versus `docs/step4_tail_before.json`. If the tail set does not improve, chunking is not earning the parent store.
- Prompt: sampled tail sentences show up in the formatted context. A section-id pin returns the parent body, not the first child row.
- `CHILD_CHUNKS=0` reingest (not a query-time flag) reproduces the golden before-file.

**If those fail:** reingest with `CHILD_CHUNKS=0`, confirm golden ranks match `docs/step4_before.json`, and do not leave chunking on. Code can stay behind the flag or be reverted; the running index must match the before file.

**Measurement (2026-11-01) — chunking left off**

Code is in place (`child_chunks.py`, parent store, parent resolve, prompt span). Default is **off**. `CHILD_CHUNKS=1 python ingest.py` turns it on. The running index was reingested with the flag off.

| Check | Before | With child chunks | Rollback (`CHILD_CHUNKS` off) |
| --- | --- | --- | --- |
| Golden hit@1 / hit@3 / hit@8 | 80% / 100% / 100% (`docs/step4_before.json`) | **50% / 100% / 100%** (`docs/step4_after.json`) | Exact per-question match to the before file (`docs/step4_rollback.json`) |
| Tail set (30 current top-8 misses) | 0% hit@8 by construction (`docs/step4_tail_before.json`) | **57% / 70% / 93%**, 2 still missing (`docs/step4_tail_after.json`) | — |
| Section-id pin `12.44.020` | — | Parent body, not a child row | — |
| Prompt span (5 tail samples) | — | Exact sentence string absent from formatted context | — |

Golden ranks that dropped: fence height 1→2, hydrant question 2→3, “shit outside” 1→2, public urination 1→3. Nothing fell out of the top 8. Ingest with chunking on: 3,580 parents, 1,265 split, 9,003 Chroma rows, window 256, almost all splits on paragraph breaks (`docs/step4_ingest_stats.json`). Clause fallback was not needed on this corpus.

Tail retrieval got better. Golden hit@1 got worse, and the sampled prompts did not contain the exact tail sentence, so the keep rule failed. Chunking stays behind the flag.

**Follow-up (same day) — parent vector + parent BM25.** Dense search kept the whole-section vector and the children, and took the closer one per section. BM25 ran on the full parent only. Result (`docs/step4b_after.json`, `docs/step4b_tail_after.json`): golden hit@1 / hit@3 / hit@8 = **60% / 90% / 100%**. Fence height returned to rank 1. The hydrant question fell to rank 4 (was 2). Tail hit@8 fell to **87%** (4 missing), below the children-only run. Parent keyword search gave back some easy ranks and gave up some of the tail lift. Still fails the keep rule. Index reingested with the flag off.

**Out of scope here:** embedding model swap, answer repair (step 5), TUI grounding (step 6).

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
| Embedding model swap inside step 4 | Step 4 only reads the current window. A model swap is a separate reingest once units and hybrid are already measured |
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

### Step 3 — Hybrid BM25 + dense RRF (2026-10-01)

**What changed**

- `hybrid_retriever.py`: BM25Okapi over Chroma units + `rrf_fuse`
- `MunicipalCodeAssistant` rebuilds BM25 on `initialize()`; default `HYBRID_SEARCH=1`
- `smart_search_code` fuses dense + BM25 after section-id pins
- `rank_bm25` added to `requirements.txt`; metrics in `docs/step3_after.json`

**Verification** (n=10 golden)

| Mode | hit@1 | hit@3 | hit@8 |
| --- | --- | --- | --- |
| Plain dense | 30% | 50% | 70% |
| Smart dense-only (step 2) | 50% | 80% | 100% |
| **Hybrid (step 3)** | **80%** | **100%** | **100%** |

Notable: fence → #1; “shit outside” → #1; hydrant natural language → #2 (was #4/#7). Ready for step 4 if finer chunking is needed, or step 5 answer-loop work.---

## How we use this doc

1. Implement **one step at a time** in the table order.  
2. After each step: smoke the app (`ingest` if needed → `main.py` ask), update status, note what changed.  
3. Sync [ARCHITECTURE.md](ARCHITECTURE.md) when behavior/contracts change.  
4. New ideas go under deferred or as a new step with an explicit dependency — don’t insert a side path that bypasses the single search or single repair loop.
