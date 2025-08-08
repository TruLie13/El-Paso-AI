#!/usr/bin/env python3
"""
Diagnostic script to find the correct import path for SelfQueryingRetriever
"""

print("Checking LangChain version and available imports...")

# Check LangChain version
try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
except:
    print("Could not determine LangChain version")

# Try different import paths
import_attempts = [
    "from langchain.retrievers.self_query.base import SelfQueryingRetriever",
    "from langchain.retrievers import SelfQueryingRetriever", 
    "from langchain.retrievers.self_query import SelfQueryingRetriever",
    "from langchain_experimental.retrievers import SelfQueryingRetriever",
    "from langchain_community.retrievers import SelfQueryingRetriever",
]

successful_import = None

for attempt in import_attempts:
    try:
        print(f"\nTrying: {attempt}")
        exec(attempt)
        print("✅ SUCCESS!")
        successful_import = attempt
        break
    except ImportError as e:
        print(f"❌ Failed: {e}")

if successful_import:
    print(f"\n🎉 Use this import: {successful_import}")
else:
    print("\n❌ Could not find SelfQueryingRetriever in any location")
    
    # Let's see what's actually available
    print("\nChecking what's available in langchain.retrievers.self_query.base:")
    try:
        import langchain.retrievers.self_query.base as base_module
        available = [attr for attr in dir(base_module) if not attr.startswith('_')]
        print("Available classes/functions:", available)
    except Exception as e:
        print(f"Error checking base module: {e}")
        
    print("\nChecking what's available in langchain.retrievers:")
    try:
        import langchain.retrievers as retrievers_module
        available = [attr for attr in dir(retrievers_module) if not attr.startswith('_')]
        print("Available in retrievers:", available)
    except Exception as e:
        print(f"Error checking retrievers module: {e}")

print("\nAlso checking your pip packages for relevant langchain packages:")
import subprocess
try:
    result = subprocess.run(['pip', 'list', '|', 'grep', 'langchain'], 
                          shell=True, capture_output=True, text=True)
    print("Installed LangChain packages:")
    print(result.stdout)
except:
    print("Could not check pip packages")