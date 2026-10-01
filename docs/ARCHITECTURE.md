# System Architecture

High-level map of how El Paso Municipal Code AI is wired today. This describes the **current code paths**, not aspirational README claims.

Planned enhancements live in [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) — add items there as we review this doc, then update this map when behavior changes.

## Big picture

Two phases: **ingest once**, then **query repeatedly**.

| Phase | Entry | Output |
| --- | --- | --- |
| Ingest | `python ingest.py` | `chroma_db/` (persistent Chroma collection) + optional `full_text_ocr.txt` cache |
| Runtime | `python main.py` | Terminal Q&A via LLM (default: local Ollama), grounded on retrieved code sections |

```mermaid
flowchart LR
  PDF["data/EP_Ordinances.pdf"] --> Ingest["ingest.py"]
  Ingest --> OCRCache["full_text_ocr.txt"]
  Ingest --> Chroma["chroma_db\n(full sections + MiniLM embeddings)"]
  Chroma --> App["main.py"]
  App --> TUI["TUIInterface"]
  TUI --> Asst["MunicipalCodeAssistant"]
  Asst --> Chroma
  Asst --> LLM["LLM (default: Ollama / llama3)\noptional: Gemini if LLM_PROVIDER=google"]
```

## Runtime components

```text
main.py
  ├── MunicipalCodeAssistant   # retrieval + LLM orchestration
  │     ├── LocalEmbeddings    # Chroma default ONNX MiniLM
  │     ├── Chroma             # vector store at chroma_db/
  │     ├── SelfQueryRetriever # optional fallback (langchain_classic)
  │     └── Chat LLM via `_create_llm()`
  │           default: Ollama (`llama3`)
  │           optional: Gemini (`LLM_PROVIDER=google`)
  └── TUIInterface             # terminal UX only
```

Supporting / secondary scripts (not on the primary path):

| File | Role |
| --- | --- |
| `load_ocr_to_db.py` | Alternate ingest into `chroma_db_final_structured` (Google embeddings; older path) |
| `debug_ask.py`, `check_databases.py`, `quick_pdf_test.py` | Diagnostics / experiments |
| `diagnostic_imports.py`, `find_retriever.py` | LangChain import hunting helpers |

## Ingestion pipeline

Primary path: `ingest.py` + `document_units.py` (step 1).

```mermaid
flowchart TD
  A["PDF pages"] --> B{"full_text_ocr.txt exists?"}
  B -->|yes| C["Load cache"]
  B -->|no| D["Parallel OCR\n(unstructured partition_pdf, multiprocessing)"]
  D --> E["Write full_text_ocr.txt"]
  E --> C
  C --> F["strip_page_chrome"]
  F --> G["Header-based segment\n(muni / Section / § / Article)"]
  G --> H["Canonicalize by section id\n+ content_type tag"]
  H --> I["Wipe chroma_db + embed LocalEmbeddings"]
  I --> J["collection: full_sections_final"]
```

What is stored per document:

- `page_content`: heading line + section body (not child chunks)
- `metadata.section`: stable unit id (e.g. `20.16.030`)
- `metadata.title`, `metadata.heading_dialect`, `metadata.content_type` (`prose` | `table`)

**Note:** Ingest **replaces** `chroma_db/` (`rmtree`), it does not append. There is still no `ParentDocumentRetriever` (that is plan step 4).

## Query pipeline

Primary path: `MunicipalCodeAssistant.ask_question`.

```mermaid
flowchart TD
  Q["User question"] --> S["smart_search_code"]
  S --> ID["Exact metadata fetch\nif section id in question"]
  S --> V["Optional topic query variants\n(hard-coded; not grown)"]
  V --> B["batch_search: similarity_search_with_score\n(dense; keep distances)"]
  ID --> M["Merge: pinned ids first,\nthen distance-sorted dense hits"]
  B --> M
  M --> F{"Fewer than 5 docs and\nSelfQuery enabled?"}
  F -->|yes| SQ["SelfQueryRetriever fallback\n(LLM builds metadata filters)"]
  F -->|no| G["Format top docs as context\n(max 8 sections, truncated)"]
  SQ --> G
  G --> L1["LLM: initial answer"]
  L1 --> C{"Answer contains\n'needs more info' phrases?"}
  C -->|no| Out["Return answer"]
  C -->|yes| T["Extract section numbers / topic terms"]
  T --> B2["batch_search again"]
  B2 --> L2["LLM: final answer with augmented context"]
  L2 --> Out
```

### Retrieval details (important)

Search is **not** hybrid BM25 + vector yet (plan step 3).

1. **Section-id pin:** if the question contains an id like `12.44.020`, fetch that unit by metadata first (`match_type=section_id`).
2. **Candidate generation:** Chroma `similarity_search_with_score` (dense); distances kept and used for ranking (lower = closer).
3. **Ranking:** distance order after id pins — **no** city/chapter-prefix heuristic bonuses.
4. **Query expansion:** existing hard-coded topic maps kept lightly (not grown); hybrid sparse search is meant to replace much of this later.
5. **Self-query:** optional fallback if first search returns few docs (policy change planned in step 5).

There is no BM25/sparse index yet. Cross-encoder re-rank was deferred; distance ranking + id pin beat the old heuristic on the golden set.

### Generation

- Default model: local **Ollama** (`llama3` via `langchain_ollama`), `http://127.0.0.1:11434`
- Optional: **Gemini** (`gemini-2.5-flash`) when `LLM_PROVIDER=google` and `GOOGLE_API_KEY` is set
- Prompt: fixed template asking for a direct YES/NO, section citation, and practical guidance
- Self-correction: string sniffing on the model output (`_quick_needs_check`); if triggered, one extra retrieval round then a second generation

## Data & external dependencies

```text
Local
  data/EP_Ordinances.pdf     source ordinances
  full_text_ocr.txt          OCR cache (gitignored)
  chroma_db/                 vector index (gitignored)
  local MiniLM (ONNX)        embeddings via chromadb DefaultEmbeddingFunction
  Ollama                     default chat LLM (answer generation + SelfQuery)

Remote (optional)
  Google Generative AI       only if LLM_PROVIDER=google
  GOOGLE_API_KEY             required only for that provider
```

Embeddings for the primary DB do **not** require Google. The default LLM path is fully local (Ollama).

## What the README overstates vs reality

| Claim | Current code |
| --- | --- |
| ParentDocumentRetriever (child chunks → parent sections) | Not used yet (plan step 4); full header units are indexed |
| Hybrid vector + keyword search | Dense retrieval + heuristic keyword-ish re-rank only (plan step 3) |
| BM25 / sparse retrieval | Not present |
| Agentic multi-step search | Light loop: search → generate → phrase check → optional second search |
| Split on every section-id mention | **Fixed (step 1):** header-based unitization via `document_units.py` |

## Extension points

If you change architecture, these are the natural seams:

1. **Ingest chunking** — `ingest.py` section split / filter logic
2. **Embeddings** — `local_embeddings.py` (must stay consistent between ingest and query)
3. **Retrieval policy** — `smart_search_code` / `batch_search` / `relevance_score`
4. **Answer loop** — `ask_question` and the `_quick_*` helpers
5. **UI** — `tui_interface.py` (swap for API/web without touching retrieval)
