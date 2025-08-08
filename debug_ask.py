import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# --- CONFIGURATION ---
# Point to our final, structured database
DB_PATH = "chroma_db"


def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        return

    print("Loading FINAL knowledge base... Please wait.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=DB_PATH,
                         embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", temperature=0)

    # First, let's check if the vectorstore has any documents
    print("Checking vectorstore contents...")
    try:
        # Test with a simple similarity search first
        test_docs = vectorstore.similarity_search("fence", k=3)
        print(f"Found {len(test_docs)} documents with simple similarity search for 'fence'")
        
        if test_docs:
            print("Sample document:")
            print(f"Section: {test_docs[0].metadata.get('section', 'N/A')}")
            print(f"Content preview: {test_docs[0].page_content[:200]}...")
        
        # Get collection info
        collection = vectorstore._collection
        print(f"Total documents in collection: {collection.count()}")
        
    except Exception as e:
        print(f"Error checking vectorstore: {e}")
        return

    # --- Self-Querying Retriever Setup ---
    metadata_field_info = [
        AttributeInfo(
            name="section",
            description="The municipal code section number, for example `7.04.010` or `18.16.020`.",
            type="string",
        ),
    ]
    document_content_description = "The text of a section of the El Paso municipal code."

    try:
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=True
        )
        use_self_query = True
    except Exception as e:
        print(f"Failed to create SelfQueryRetriever: {e}")
        print("Falling back to simple similarity search...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        use_self_query = False
    
    # --- End Setup ---

    print("Knowledge base loaded. You can now ask questions.")
    print("Type 'exit' or 'quit' to end the session.")
    if use_self_query:
        print("Using SelfQueryRetriever with structured queries.")
    else:
        print("Using basic similarity search.")

    while True:
        try:
            question = input("\n> Ask a question: ")
            if question.lower() in ["exit", "quit"]:
                break

            print("\nSearching...")
            
            if use_self_query:
                retrieved_docs = retriever.invoke(question)
            else:
                retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                print("Could not find any relevant documents.")
                # Try a broader search
                print("Trying broader search...")
                broader_docs = vectorstore.similarity_search(question, k=5)
                if broader_docs:
                    print(f"Found {len(broader_docs)} documents with broader search:")
                    for i, doc in enumerate(broader_docs[:2]):
                        print(f"\n--- Document {i+1} - Section {doc.metadata.get('section', 'N/A')} ---")
                        print(f"{doc.page_content[:300]}...")
                continue

            print(f"\n--- Found {len(retrieved_docs)} Matching Section(s) ---")
            for i, doc in enumerate(retrieved_docs):
                print(f"\n--- Document {i+1} - Section {doc.metadata.get('section', 'N/A')} ---")
                print(f"{doc.page_content[:700]}...")
            print("-" * 50)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()

    print("\nSession ended.")


if __name__ == "__main__":
    main()