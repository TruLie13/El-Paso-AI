# El Paso Municipal Code AI Assistant 
A multi-step, agentic RAG pipeline with a custom scoring retriever for accurately answering complex questions about the El Paso, TX municipal code.

*** A simplified version of this project, adapted for a bootcamp curriculum with a more streamlined architecture and using Ollama for local inference, is also available here: https://github.com/TruLie13/municipal-ai ***

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Implementation](#implementation)
- [Results](#results)
- [Cost & Scaling](#cost--scaling)
- [Getting Started](#getting-started)
- [Lessons Learned](#lessons-learned)
- [Future Improvements](#future-improvements)

<img width="639" height="990" alt="image" src="https://github.com/user-attachments/assets/6d7b398d-ecf4-4033-a609-84d204e980d2" />

<img width="639" height="990" alt="image" src="https://github.com/user-attachments/assets/a5371461-6c88-4a88-a6af-f4aafc5c1bef" />

## Project Overview
This project is an advanced Question-Answering system designed to navigate the dense and complex El Paso municipal code. It goes beyond standard RAG by implementing a multi-step, agentic search process and a custom relevance scoring algorithm to provide accurate, context-aware, and cited answers to citizen inquiries, making complex legal text accessible to the public.

## Problem Statement
Municipal codes are often hundreds of pages long, written in dense legalese, and poorly indexed for public use. This makes it incredibly difficult for citizens and business owners to find definitive answers to simple questions (e.g., "How tall can my fence be?" or "What are the rules for street parking?"), leading to frustration, non-compliance, and an increased burden on city officials.

## Architecture
The system is built on a sophisticated RAG pipeline that uses both pre-defined search strategies and agentic reasoning to find the best possible answer.

**Data Flow:**

1. **Ingestion:** A large PDF of the municipal code is processed in parallel using Python's multiprocessing. The raw text is extracted using OCR, cleaned, and cached to a text file for efficiency.

2. **Indexing:** The cleaned text is parsed using Regex to identify parent sections. A `ParentDocumentRetriever` then splits these sections into smaller child chunks for embedding, storing the parents in an in-memory docstore and the child embeddings in a persistent ChromaDB vector store.

3. **Querying:** A user question initiates a multi-step process orchestrated by the MunicipalCodeAssistant class:
   - **Smart Search:** A custom `smart_search_code` function generates multiple query variations and retrieves an initial set of documents. It then applies a custom scoring algorithm based on word matches, section numbers, and legal keywords to re-rank the results for relevance.
   - **Self-Query Fallback:** If the initial search yields few results, a SelfQueryRetriever is used as a fallback to translate the question into a structured query with metadata filters.
   - **Initial Generation:** The top-ranked documents are passed to a Google Gemini LLM to generate an initial answer.
   - **Self-Correction Loop:** The assistant analyzes its own answer. If it determines more information is needed, it extracts new search terms and performs a second, targeted retrieval to augment the context.
   - **Final Generation:** A final, comprehensive answer is generated from the augmented context.

## Implementation
- **Parallel Ingestion:** Utilized Python's multiprocessing to dramatically speed up the initial, time-consuming OCR and text extraction from the source PDF.

- **Advanced Retrieval:** Implemented a `ParentDocumentRetriever` to combine the precision of small chunk search with the context of full document retrieval.

- **Custom Relevance Scoring:** Developed a `smart_search_code` function that goes beyond basic vector similarity. It generates multiple queries and re-ranks results using a custom algorithm that boosts scores based on keyword matches and the relevance of specific municipal code titles.

```python
def relevance_score(doc):
    # ... word matching, section bonuses, keyword bonuses ...
    return word_match_score + section_bonus + quality_bonus

all_docs.sort(key=relevance_score, reverse=True)
```

- **Agentic Search Loop:** The main `ask_question` method functions as a simple agent. It performs a search, generates an answer, checks its own answer for completeness, and can trigger a secondary, targeted search to refine the result.

```python
# Simplified logic of the agentic loop
initial_docs = self.smart_search_code(question)
answer = self.summary_chain.invoke({"context": initial_docs, "question": question})

if self._quick_needs_check(answer):
    additional_terms = self._extract_quick_search_terms(answer)
    extra_docs = self.batch_search(additional_terms)
    # ... generate final answer with combined context
```

## Results
The primary success metric is the qualitative improvement in answer quality over a standard RAG implementation. The multi-step, re-ranking, and self-correction process consistently retrieves more relevant documents for complex or ambiguous queries.

**Example Scenario:**

- **Question:** "Can I park my car near a fire hydrant?"
- **Standard RAG Result:** Often retrieves fragmented sections about "fire access roads" or "parking prohibitions" without a specific distance, leading to a vague answer.
- **This Project's Result:** The multi-query search and relevance scoring successfully locate the specific section defining fire lane obstructions. The final answer is direct, cites the code, and provides the specific distance (e.g., "No, you cannot park within 15 feet of a fire hydrant as per section X.Y.Z, which defines this area as a fire lane.").

## Cost & Scaling
**Cost:** The application primarily uses a local vector database (`ChromaDB`) and local processing. The only external cost is API calls to the Google Generative AI (Gemini) model for the generation steps. Caching and efficient, targeted retrieval minimize the number and size of these calls.

**Scaling:** The most computationally expensive step, the initial PDF ingestion, is scaled across multiple CPU cores using multiprocessing. The retrieval logic is optimized for performance on a single machine.

## Getting Started
### Prerequisites:
- Python 3.11+
- An environment variable GOOGLE_API_KEY set in a .env file.

### Installation:
1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd <repository-name>`
3. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies: `pip install -r requirements.txt`

### Running the Application:
1. Place the source PDF in the `/data` directory.
2. update ```` PDF_PATH = "data/EP_Ordinances.pdf" ```` to match your PDF filename in `python ingest.py`
3. Run the ingestion script to build the database: `python ingest.py`
4. Run the main application: `python main.py`

## Lessons Learned
- Standard RAG is often insufficient for dense, structured documents like legal code. A simple vector search does not guarantee retrieval of the most relevant context.
- Custom relevance scoring and multi-query strategies are critical for improving retrieval accuracy in specialized domains.
- Agentic, multi-step reasoning (search -> generate -> check -> re-search) produces significantly more comprehensive and accurate answers than a single-pass RAG chain.

## Future Improvements
- Deploy the MunicipalCodeAssistant as a REST API using FastAPI.
- Build a simple web-based user interface using Streamlit or Gradio.
- Implement an automated pipeline to update the vector database when the source municipal code PDF is updated.
- Add monitoring for model drift and retrieval quality.

