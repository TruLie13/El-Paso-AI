import os
import glob
from pathlib import Path

def main():
    print("Checking for PDF files and processing evidence...")
    
    # Look for PDF files
    pdf_files = []
    for pattern in ['*.pdf', '**/*.pdf']:
        pdf_files.extend(glob.glob(pattern, recursive=True))
    
    if pdf_files:
        print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            size = os.path.getsize(pdf)
            size_mb = size / (1024 * 1024)
            print(f"   - {pdf} ({size_mb:.2f} MB)")
    else:
        print("\n❌ No PDF files found in current directory!")
    
    # Look for any ingestion/processing scripts
    print("\n🔍 Looking for processing scripts...")
    script_patterns = ['*ingest*', '*process*', '*create*', '*build*', '*load*']
    scripts = []
    
    for pattern in script_patterns:
        scripts.extend(glob.glob(f"{pattern}.py"))
    
    # Also check common script names
    common_names = ['main.py', 'setup.py', 'build_db.py', 'index.py']
    for name in common_names:
        if os.path.exists(name) and name not in scripts:
            scripts.append(name)
    
    if scripts:
        print("Found potential processing scripts:")
        for script in scripts:
            print(f"   - {script}")
    else:
        print("No obvious processing scripts found")
    
    # Check for any log files or output
    print("\n📝 Checking for log files...")
    log_patterns = ['*.log', '*.out', '*.txt']
    logs = []
    for pattern in log_patterns:
        logs.extend(glob.glob(pattern))
    
    if logs:
        print("Found potential log files:")
        for log in logs[:5]:  # Show first 5
            size = os.path.getsize(log)
            print(f"   - {log} ({size} bytes)")
    
    # Check directory structure
    print(f"\n📁 Current directory: {os.getcwd()}")
    print("Directory contents:")
    items = sorted(os.listdir('.'))
    for item in items:
        if os.path.isdir(item):
            print(f"   📁 {item}/")
        else:
            size = os.path.getsize(item)
            if size > 1024:  # Show files > 1KB
                size_kb = size / 1024
                print(f"   📄 {item} ({size_kb:.1f} KB)")
    
    # Try to import and test basic PDF processing
    print("\n🧪 Testing PDF processing capabilities...")
    try:
        import PyPDF2
        print("✅ PyPDF2 available")
    except ImportError:
        print("❌ PyPDF2 not available")
    
    try:
        import pymupdf  # fitz
        print("✅ PyMuPDF available")
    except ImportError:
        print("❌ PyMuPDF not available")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        print("✅ LangChain PDF loader available")
    except ImportError:
        print("❌ LangChain PDF loader not available")
    
    if pdf_files:
        pdf_file = pdf_files[0]
        print(f"\n🔬 Quick test of PDF: {pdf_file}")
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()
            print(f"✅ Successfully loaded PDF: {len(pages)} pages")
            if pages:
                print(f"   First page preview: {pages[0].page_content[:200]}...")
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")

if __name__ == "__main__":
    main()