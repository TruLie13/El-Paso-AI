import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm

# Configuration
OCR_TEXT_PATH = "full_text_ocr.txt"
DB_PATH = "chroma_db_final_structured"
MAX_WORKERS = 6  # Use 6 out of 8 cores to avoid overwhelming system
BATCH_SIZE = 50  # Smaller batches for better parallelization

# Thread-safe counter
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def parse_municipal_sections_fast(text):
    """Fast parsing of municipal code sections"""
    print("🚀 Fast parsing municipal code sections...")
    
    section_pattern = r'(\d+\.\d+\.\d+)'
    sections = []
    
    # Split text into larger chunks first for parallel processing
    lines = text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        section_match = re.match(section_pattern, line)
        
        if section_match:
            if current_section and current_content:
                content = '\n'.join(current_content).strip()
                if len(content) > 50:
                    sections.append({
                        'section': current_section,
                        'content': content
                    })
            
            current_section = section_match.group(1)
            current_content = [line]
        else:
            if current_content:
                current_content.append(line)
    
    # Last section
    if current_section and current_content:
        content = '\n'.join(current_content).strip()
        if len(content) > 50:
            sections.append({
                'section': current_section,
                'content': content
            })
    
    return sections

def create_document_batch(section_batch):
    """Create a batch of documents - this runs in parallel"""
    documents = []
    for section_data in section_batch:
        doc = Document(
            page_content=section_data['content'],
            metadata={
                'section': section_data['section'],
                'source': 'EP_Ordinances.pdf'
            }
        )
        documents.append(doc)
    return documents

def add_batch_to_vectorstore(vectorstore, document_batch, batch_num, counter, pbar):
    """Add a batch of documents to vectorstore - thread-safe"""
    try:
        vectorstore.add_documents(document_batch)
        count = counter.increment()
        pbar.update(1)
        return f"✅ Batch {batch_num} ({len(document_batch)} docs) added successfully"
    except Exception as e:
        pbar.update(1)
        return f"❌ Batch {batch_num} failed: {e}"

def chunk_list(lst, chunk_size):
    """Split list into chunks"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        return
    
    if not os.path.exists(OCR_TEXT_PATH):
        print(f"Error: OCR text file not found at {OCR_TEXT_PATH}")
        return
    
    print(f"🔥 TURBO MODE: Loading OCR text from {OCR_TEXT_PATH} with {MAX_WORKERS} threads...")
    start_time = time.time()
    
    # Load OCR text
    with open(OCR_TEXT_PATH, 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    print(f"📖 Loaded {len(ocr_text):,} characters")
    
    # Parse sections
    sections = parse_municipal_sections_fast(ocr_text)
    print(f"📑 Parsed {len(sections)} sections")
    
    if len(sections) < 10:
        print("⚠️  Few sections found, using fallback chunking...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_text(ocr_text)
        sections = []
        for i, chunk in enumerate(chunks):
            section_match = re.search(r'(\d+\.\d+\.\d+)', chunk)
            section = section_match.group(1) if section_match else f"chunk_{i:04d}"
            sections.append({'section': section, 'content': chunk})
        print(f"📝 Created {len(sections)} chunks")
    
    # Create documents in parallel batches
    print(f"🏭 Creating documents using {MAX_WORKERS} workers...")
    section_batches = list(chunk_list(sections, BATCH_SIZE))
    
    all_documents = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(section_batches), desc="Creating docs") as pbar:
            future_to_batch = {
                executor.submit(create_document_batch, batch): i 
                for i, batch in enumerate(section_batches)
            }
            
            for future in as_completed(future_to_batch):
                documents = future.result()
                all_documents.extend(documents)
                pbar.update(1)
    
    print(f"📄 Created {len(all_documents)} total documents")
    
    # Show sample
    if all_documents:
        print(f"📋 Sample document:")
        print(f"   Section: {all_documents[0].metadata.get('section')}")
        print(f"   Content: {all_documents[0].page_content[:200]}...")
    
    # Initialize database
    print(f"🗄️  Initializing database at {DB_PATH}...")
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # Add documents to database in parallel
    print(f"⚡ Adding {len(all_documents)} documents to database with {MAX_WORKERS} threads...")
    document_batches = list(chunk_list(all_documents, BATCH_SIZE))
    total_batches = len(document_batches)
    
    counter = ThreadSafeCounter()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=total_batches, desc="Adding to DB") as pbar:
            futures = []
            
            for i, batch in enumerate(document_batches):
                future = executor.submit(
                    add_batch_to_vectorstore, 
                    vectorstore, batch, i+1, counter, pbar
                )
                futures.append(future)
            
            # Wait for all to complete and collect results
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    # Show results
    successful = len([r for r in results if "✅" in r])
    failed = len([r for r in results if "❌" in r])
    
    print(f"\n📊 Results: {successful} successful, {failed} failed")
    if failed > 0:
        print("❌ Failed batches:")
        for result in results:
            if "❌" in result:
                print(f"   {result}")
    
    # Verify database
    print("🔍 Verifying database...")
    try:
        collection = vectorstore._collection
        count = collection.count()
        print(f"✅ Database has {count:,} documents!")
        
        # Quick test
        test_results = vectorstore.similarity_search("fence height", k=3)
        print(f"🔎 Test search for 'fence height': {len(test_results)} results")
        
        if test_results:
            result = test_results[0]
            print(f"   📋 Sample: Section {result.metadata.get('section')} | {result.page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n🎉 COMPLETE! Total time: {elapsed:.1f} seconds")
    print(f"📍 Database ready at: {DB_PATH}")

if __name__ == "__main__":
    main()