import os
import shutil
import time
from dotenv import load_dotenv
from tqdm import tqdm

import fitz  # PyMuPDF
from langchain_chroma import Chroma
from local_embeddings import LocalEmbeddings
from document_units import segment_into_units, structural_stats

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

    # Stage C: wipe + rebuild Chroma (replace, not append)
    print("\nSetting up ChromaDB with local embeddings (wipe + rebuild)...")
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    embeddings = LocalEmbeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    print("\nAdding documents in batches... (Embedding and Indexing)")
    batch_size = 50
    for i in tqdm(range(0, len(documents), batch_size), desc="Adding Batches"):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)

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
