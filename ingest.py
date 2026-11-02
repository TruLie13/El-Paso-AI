import json
import os
import shutil
import time
from dotenv import load_dotenv
from tqdm import tqdm

import fitz  # PyMuPDF
from langchain_chroma import Chroma

from local_embeddings import LocalEmbeddings
from document_units import segment_into_units, structural_stats
from child_chunks import expand_units_for_index, write_parent_store

# --- CONFIGURATION ---
PDF_PATH = "data/EP_Ordinances.pdf"
OCR_TEXT_CACHE = "full_text_ocr.txt"  # File to save/load OCR results
DB_PATH = "chroma_db"
COLLECTION_NAME = "full_sections_final"


def get_ocr_text():
    """
    Performs parallel OCR and saves the result to a cache file.
    If the cache file already exists, it loads from there instead.
    """
    if os.path.exists(OCR_TEXT_CACHE):
        print(f"Found cached OCR text. Loading from '{OCR_TEXT_CACHE}'...")
        with open(OCR_TEXT_CACHE, "r", encoding="utf-8") as f:
            return f.read()

    # If cache doesn't exist, run the long OCR process
    import math
    import multiprocessing

    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    doc.close()

    num_cores = multiprocessing.cpu_count()
    print(
        f"Starting parallel OCR with {num_cores} CPU cores. This will take hours..."
    )

    chunk_size = math.ceil(total_pages / num_cores)
    page_chunks = [
        (i * chunk_size + 1, min((i + 1) * chunk_size, total_pages))
        for i in range(num_cores)
    ]

    start_time = time.time()
    all_text_chunks = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        with tqdm(total=len(page_chunks), desc="Parallel OCR Progress") as pbar:
            for text_list in pool.imap_unordered(process_page_chunk, page_chunks):
                all_text_chunks.extend(text_list)
                pbar.update(1)

    print(
        f"\nParallel OCR finished in {time.time() - start_time:.2f} seconds."
    )

    full_text = "\n".join(all_text_chunks)

    print(f"Saving OCR text to cache file: '{OCR_TEXT_CACHE}'")
    with open(OCR_TEXT_CACHE, "w", encoding="utf-8") as f:
        f.write(full_text)

    return full_text


def process_page_chunk(page_chunk):
    # This is the worker function for the multiprocessing pool
    from unstructured.partition.pdf import partition_pdf

    start_page, end_page = page_chunk
    try:
        elements = partition_pdf(
            filename=PDF_PATH,
            strategy="ocr_only",
            starting_page_number=start_page,
            ending_page_number=end_page,
        )
        return [str(el) for el in elements]
    except Exception as e:
        print(f"Error in worker for pages {start_page}-{end_page}: {e}")
        return []


def main():
    load_dotenv()

    # Stage A: Get the full text, either from cache or by running OCR
    full_text = get_ocr_text()

    # Stage B (step 1): chrome strip → header split → canonicalize → tag
    print("\nStep 1: Building document units (gov/corporate heading segmentation)...")
    documents = segment_into_units(full_text)
    stats = structural_stats(full_text, documents)
    print(f"  Indexed units: {stats['indexed_units']}")
    print(f"  Header matches: {stats['header_matches']}")
    print(f"  Unique header ids: {stats['unique_header_ids']}")
    print(f"  Units with chrome: {stats['units_with_chrome']}")
    print(f"  prose={stats['prose_units']} table={stats['table_units']}")
    print(f"  dialects: {stats['dialects']}")
    print(f"  avg unit chars: {stats['avg_body_chars']}")

    if not documents:
        raise RuntimeError("No document units produced — aborting ingest.")

    # Stage C: child chunks only when a prose parent exceeds the embedder window.
    print("\nStep 4: Deciding child chunks from the live embedder window...")
    embeddings = LocalEmbeddings()
    index_docs, parents, chunk_stats = expand_units_for_index(documents, embeddings)
    print(
        f"  model={chunk_stats['model_name']} max_tokens={chunk_stats['max_tokens']} "
        f"enabled={chunk_stats['enabled']}"
    )
    print(
        f"  parents={chunk_stats['parents']} stored_whole={chunk_stats['stored_whole']} "
        f"split={chunk_stats['split_parents']} tables_whole={chunk_stats['tables_left_whole']}"
    )
    print(
        f"  chroma_docs={chunk_stats['chroma_docs']} "
        f"clause_sections={chunk_stats['sections_using_clause']} "
        f"sentence_sections={chunk_stats['sections_using_sentence']} "
        f"hard_sections={chunk_stats['sections_using_hard']}"
    )
    stats_name = (
        "docs/step4b_ingest_stats.json"
        if chunk_stats["enabled"]
        else "docs/step4_rollback_ingest_stats.json"
    )
    os.makedirs("docs", exist_ok=True)
    with open(stats_name, "w", encoding="utf-8") as handle:
        json.dump(chunk_stats, handle, indent=2)
    print(f"  wrote {stats_name}")

    # Wipe + rebuild Chroma (replace, not append), then write the parent store
    # into the same directory so the next wipe removes both.
    print("\nSetting up ChromaDB with local embeddings (wipe + rebuild)...")
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    print("\nAdding documents in batches... (Embedding and Indexing)")
    batch_size = 50
    for i in tqdm(range(0, len(index_docs), batch_size), desc="Adding Batches"):
        batch = index_docs[i : i + batch_size]
        vectorstore.add_documents(batch)

    parent_path = write_parent_store(DB_PATH, parents)
    print(f"Parent store: {parent_path} ({len(parents)} sections)")

    print("\n--- FINAL INGESTION COMPLETE ---")
    print(f"Database ready at: {DB_PATH}")
    print(f"Collection count: {vectorstore._collection.count()}")

    # Quick retrieval sanity check
    hits = vectorstore.similarity_search("fence height", k=2)
    if hits:
        print(
            f"Test search OK — top hit section: {hits[0].metadata.get('section')} "
            f"title={hits[0].metadata.get('title')!r}"
        )


if __name__ == "__main__":
    main()
