import os
import glob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def check_database(db_path):
    """Check a single database for content"""
    try:
        if not os.path.exists(db_path):
            return f"{db_path}: Directory does not exist"
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        collection = vectorstore._collection
        count = collection.count()
        
        if count > 0:
            # Get a sample document
            sample_docs = vectorstore.similarity_search("code", k=1)
            if sample_docs:
                sample_metadata = sample_docs[0].metadata
                sample_content = sample_docs[0].page_content[:100]
                return f"{db_path}: {count} documents | Sample: {sample_metadata} | Content: {sample_content}..."
            else:
                return f"{db_path}: {count} documents (but search returned nothing)"
        else:
            return f"{db_path}: EMPTY (0 documents)"
            
    except Exception as e:
        return f"{db_path}: ERROR - {e}"

def main():
    load_dotenv()
    
    print("Searching for Chroma databases...")
    
    # Look for chroma directories
    chroma_dirs = []
    
    # Check current directory
    for item in os.listdir('.'):
        if 'chroma' in item.lower() and os.path.isdir(item):
            chroma_dirs.append(item)
    
    # Also check some common patterns
    patterns = [
        'chroma_db*',
        '*chroma*',
        'db*',
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.isdir(match) and match not in chroma_dirs:
                chroma_dirs.append(match)
    
    if not chroma_dirs:
        print("No Chroma database directories found!")
        print("Current directory contents:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  📁 {item}")
        return
    
    print(f"Found {len(chroma_dirs)} potential database directories:")
    print("-" * 60)
    
    populated_dbs = []
    
    for db_path in sorted(chroma_dirs):
        result = check_database(db_path)
        print(result)
        
        if "documents" in result and "EMPTY" not in result and "ERROR" not in result:
            populated_dbs.append(db_path)
    
    print("-" * 60)
    
    if populated_dbs:
        print(f"\n✅ Found {len(populated_dbs)} populated database(s):")
        for db in populated_dbs:
            print(f"   - {db}")
        
        print(f"\n💡 To use the populated database, update DB_PATH in your ask.py to:")
        print(f'   DB_PATH = "{populated_dbs[0]}"')
    else:
        print("\n❌ No populated databases found!")
        print("\n🔧 You may need to:")
        print("   1. Re-run your data ingestion script")
        print("   2. Check if your PDF processing completed successfully")
        print("   3. Verify the database creation process worked")

if __name__ == "__main__":
    main()