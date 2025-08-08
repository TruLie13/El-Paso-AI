import os
import re
import time
from dotenv import load_dotenv
from tqdm import tqdm

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
PDF_PATH = "data/EP_Ordinances.pdf"
OCR_TEXT_CACHE = "full_text_ocr.txt"  # File to save/load OCR results
DB_PATH = "chroma_db"


def get_ocr_text():
    """
    Performs parallel OCR and saves the result to a cache file.
    If the cache file already exists, it loads from there instead.
    """
    if os.path.exists(OCR_TEXT_CACHE):
        print(f"Found cached OCR text. Loading from '{OCR_TEXT_CACHE}'...")
        with open(OCR_TEXT_CACHE, 'r', encoding='utf-8') as f:
            return f.read()

    # If cache doesn't exist, run the long OCR process
    import multiprocessing
    import math

    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    doc.close()

    num_cores = multiprocessing.cpu_count()
    print(
        f"Starting parallel OCR with {num_cores} CPU cores. This will take hours...")

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
        f"\nParallel OCR finished in {time.time() - start_time:.2f} seconds.")

    full_text = "\n".join(all_text_chunks)

    print(f"Saving OCR text to cache file: '{OCR_TEXT_CACHE}'")
    with open(OCR_TEXT_CACHE, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return full_text


def process_page_chunk(page_chunk):
    # This is the worker function for the multiprocessing pool
    from unstructured.partition.pdf import partition_pdf
    start_page, end_page = page_chunk
    try:
        elements = partition_pdf(
            filename=PDF_PATH, strategy="ocr_only",
            starting_page_number=start_page, ending_page_number=end_page
        )
        return [str(el) for el in elements]
    except Exception as e:
        print(f"Error in worker for pages {start_page}-{end_page}: {e}")
        return []


def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        return

    # Step 1: Get the full text, either from cache or by running OCR
    full_text = get_ocr_text()

    # Step 2: Intelligently split the text by section number
    print("\nStep 2: Splitting text by Municipal Code Sections...")
    regex = r"(\d{1,2}\.\d{1,2}\.\d{1,3})"
    split_text = re.split(regex, full_text)

    parent_documents = []
    for i in range(1, len(split_text), 2):
        section_number = split_text[i]
        content = split_text[i+1]
        parent_documents.append(
            Document(page_content=content, metadata={
                     "section": section_number})
        )

    print(f"Found {len(parent_documents)} potential sections.")

    # Step 3: Filter out small, invalid sections
    print("\nStep 3: Filtering out invalid sections...")
    min_length = 100  # A section must have at least 100 characters
    filtered_documents = [doc for doc in parent_documents if len(
        doc.page_content) > min_length]
    print(f"Kept {len(filtered_documents)} valid sections after filtering.")

    # Step 4: Set up the retriever
    print("\nStep 4: Setting up the Parent Document Retriever...")
    docstore = InMemoryStore()
    vectorstore = Chroma(
        collection_name="full_sections_final",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"),
        persist_directory=DB_PATH,
    )
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, docstore=docstore, child_splitter=child_splitter
    )

    # Step 5: Add documents in batches
    print("\nStep 5: Adding documents in batches... (Embedding and Indexing)")
    batch_size = 50
    for i in tqdm(range(0, len(filtered_documents), batch_size), desc="Adding Batches"):
        batch = filtered_documents[i:i + batch_size]
        retriever.add_documents(batch, ids=None)

    vectorstore.persist()
    print("\n--- FINAL INGESTION COMPLETE ---")


if __name__ == "__main__":
    main()
