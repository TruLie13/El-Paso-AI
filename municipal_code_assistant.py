import os
import re
from dotenv import load_dotenv
from datetime import datetime
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from local_embeddings import LocalEmbeddings
from hybrid_retriever import BM25Index, rrf_fuse

# Collection name must match ingest.py
CODE_COLLECTION = "full_sections_final"

# Hybrid (BM25 + dense RRF). Set HYBRID_SEARCH=0 to use dense-only P0 path.
def _env_flag(name: str, default: bool = True) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")

# Municipal + common gov heading ids appearing in user questions
_SECTION_ID_RE = re.compile(
    r"\b(\d{1,2}\.\d{1,2}\.\d{1,3})\b"
    r"|(?:Section\s+(\d+(?:\.\d+){0,4}))"
    r"|(?:§\s*(\d+(?:\.\d+){0,4}))",
    re.I,
)


class MunicipalCodeAssistant:
    """Optimized assistant for querying El Paso municipal code"""
    
    def __init__(self, db_path="chroma_db"):
        self.db_path = db_path
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.llm_provider = None
        self.retriever = None
        self.summary_chain = None
        self.use_self_query = False
        self.bm25_index: BM25Index | None = None
        self.use_hybrid = _env_flag("HYBRID_SEARCH", default=True)

    def _create_llm(self):
        """Default to local Ollama; set LLM_PROVIDER=google to use Gemini instead."""
        provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()

        if provider in ("google", "gemini"):
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError(
                    "LLM_PROVIDER=google but GOOGLE_API_KEY is not set"
                )
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
            self.llm_provider = f"google:{model}"
            return ChatGoogleGenerativeAI(model=model, temperature=0.1)

        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.llm_provider = f"ollama:{model}"
        return ChatOllama(model=model, base_url=base_url, temperature=0.1)
        
    def initialize(self):
        """Initialize all AI components"""
        load_dotenv()

        # Must match ingest.py embeddings so query vectors align with the DB
        self.embeddings = LocalEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            collection_name=CODE_COLLECTION,
            embedding_function=self.embeddings,
        )
        self.llm = self._create_llm()
        
        # Setup self-query retriever
        metadata_field_info = [
            AttributeInfo(
                name="section",
                description="The municipal code section number, for example `7.04.010` or `18.16.020`.",
                type="string",
            ),
        ]
        document_content_description = "The text of a section of the El Paso municipal code."
        
        try:
            self.retriever = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                document_contents=document_content_description,
                metadata_field_info=metadata_field_info,
                verbose=False
            )
            self.use_self_query = True
        except Exception:
            self.use_self_query = False
        
        self.summary_chain = self._create_summary_chain()
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        """Rebuild sparse index from current Chroma units (same corpus as dense)."""
        self.bm25_index = BM25Index()
        if not self.vectorstore:
            return
        try:
            n = self.bm25_index.build_from_vectorstore(self.vectorstore)
            if n == 0:
                self.bm25_index = None
        except Exception:
            self.bm25_index = None
    
    def _create_summary_chain(self):
        """Create a streamlined chain for summarizing search results"""
        summary_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an expert on El Paso municipal code. Based on these code sections, provide a definitive, practical answer.

QUESTION: {question}

CODE SECTIONS:
{context}

INSTRUCTIONS:
- Lead with a clear, direct answer in plain language (do not force a YES/NO opener)
- Cite the specific section that applies
- If you find a relevant section (like public indecency, disorderly conduct, etc.), apply it confidently
- Include penalties/consequences if mentioned in the sections
- Be authoritative - if the law clearly applies, state it definitively
- Only mention "additional information needed" if truly critical information is missing
- Focus on practical guidance

ANSWER:"""
        )
        return summary_prompt | self.llm

    @staticmethod
    def extract_section_ids(question: str) -> list[str]:
        """Pull municipal / Section / § ids from a user question."""
        found: list[str] = []
        seen: set[str] = set()
        for m in _SECTION_ID_RE.finditer(question):
            sid = next(g for g in m.groups() if g)
            if sid not in seen:
                seen.add(sid)
                found.append(sid)
        return found

    def get_docs_by_section(self, section_id: str) -> list[Document]:
        """Exact metadata lookup (no embedding)."""
        if not self.vectorstore:
            return []
        try:
            raw = self.vectorstore.get(
                where={"section": section_id},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []
        docs: list[Document] = []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        for content, meta in zip(documents, metadatas):
            if content is None:
                continue
            metadata = dict(meta or {})
            metadata.setdefault("section", section_id)
            metadata["distance"] = 0.0
            metadata["match_type"] = "section_id"
            docs.append(Document(page_content=content, metadata=metadata))
        return docs
    
    @lru_cache(maxsize=100)
    def _get_cached_search_variations(self, question_lower_hash):
        """Cached search variations to avoid recomputation"""
        # Convert hash back to detect patterns (simplified)
        variations = []
        
        if any(word in str(question_lower_hash) for word in ['fence', 'wall', 'height']):
            variations = [
                "residential fence height limits",
                "20.16.030 fence height",
                "fence height zoning"
            ]
        elif any(word in str(question_lower_hash) for word in ['permit', 'build']):
            variations = [
                "building permit requirements",
                "fence permit"
            ]
        
        return variations
    
    def batch_search(self, queries, k_per_query=2):
        """
        Dense search keeping Chroma distances.

        Dedupes by section id, keeping the best (lowest) distance per section.
        Attaches metadata['distance'] for ranking.
        """
        best: dict[str, tuple[Document, float]] = {}

        for query in queries:
            try:
                pairs = self.vectorstore.similarity_search_with_score(
                    query, k=k_per_query
                )
            except Exception:
                continue
            for doc, distance in pairs:
                section = doc.metadata.get("section") or str(
                    hash(doc.page_content[:100])
                )
                dist = float(distance)
                prev = best.get(section)
                if prev is None or dist < prev[1]:
                    meta = dict(doc.metadata)
                    meta["distance"] = dist
                    meta.setdefault("match_type", "dense")
                    best[section] = (
                        Document(page_content=doc.page_content, metadata=meta),
                        dist,
                    )

        # Sort by ascending distance (Chroma: lower = closer)
        ordered = sorted(best.values(), key=lambda t: t[1])
        return [doc for doc, _ in ordered]
    
    def smart_search_code(self, question, k=10):
        """
        Hybrid retrieval (step 3): section-id pin + dense + BM25 fused with RRF.

        Set HYBRID_SEARCH=0 for dense-only (step 2) behavior.
        """
        pinned: list[Document] = []
        seen: set[str] = set()
        for sid in self.extract_section_ids(question):
            for doc in self.get_docs_by_section(sid):
                section = doc.metadata.get("section")
                if section and section not in seen:
                    seen.add(section)
                    pinned.append(doc)

        search_queries = [question]
        question_lower = question.lower()

        # Pre-defined high-value search patterns (unchanged set; do not grow)
        if any(
            word in question_lower
            for word in [
                "shit",
                "defecate",
                "defecation",
                "urinate",
                "urination",
                "pee",
                "bathroom",
                "toilet",
                "public restroom",
                "calls of nature",
            ]
        ):
            search_queries.extend(
                [
                    "public urination defecation prohibited",
                    "indecent conduct public decency",
                    "disorderly conduct public behavior",
                    "public health sanitation violations",
                    "nuisance public place bathroom",
                ]
            )

        if any(word in question_lower for word in ["fence", "wall", "height"]):
            search_queries.extend(
                [
                    "residential fence height 20.16.030",
                    "fence screening wall residential",
                ]
            )

        if any(word in question_lower for word in ["animal", "tiger", "pet", "dog"]):
            search_queries.extend(
                [
                    "animal control dangerous animals",
                    "exotic animal prohibition",
                ]
            )

        if any(
            word in question_lower for word in ["business", "commercial", "store"]
        ):
            search_queries.extend(
                [
                    "business license commercial",
                    "zoning commercial activity",
                ]
            )

        search_queries = search_queries[:6]
        k_per = max(3, (k * 2) // max(1, len(search_queries)))
        dense_docs = self.batch_search(search_queries, k_per_query=k_per)

        use_hybrid = (
            self.use_hybrid
            and self.bm25_index is not None
            and self.bm25_index.ready
        )

        if not use_hybrid:
            merged = list(pinned)
            for doc in dense_docs:
                section = doc.metadata.get("section")
                if section in seen:
                    continue
                seen.add(section)
                merged.append(doc)
            return merged[:k]

        # BM25 over original question + same light expansions (dedupe queries)
        bm25_pool_k = max(20, k * 3)
        bm25_by_section: dict[str, Document] = {}
        bm25_ranking: list[str] = []
        for q in search_queries:
            for hit in self.bm25_index.search(q, k=bm25_pool_k):
                section = hit.document.metadata.get("section")
                if not section or section in bm25_by_section:
                    continue
                bm25_by_section[section] = hit.document
                bm25_ranking.append(section)

        dense_by_section = {
            d.metadata.get("section"): d
            for d in dense_docs
            if d.metadata.get("section")
        }
        dense_ranking = [d.metadata.get("section") for d in dense_docs]

        fused = rrf_fuse([dense_ranking, bm25_ranking], rrf_k=60)
        doc_lookup = {**bm25_by_section, **dense_by_section}

        merged = list(pinned)
        for section, rrf_score in fused:
            if section in seen:
                continue
            doc = doc_lookup.get(section)
            if not doc:
                # Prefer BM25 store / chroma exact get as fallback
                if self.bm25_index:
                    doc = self.bm25_index.get_by_section(section)
                if not doc:
                    got = self.get_docs_by_section(section)
                    doc = got[0] if got else None
            if not doc:
                continue
            meta = dict(doc.metadata)
            meta["rrf_score"] = float(rrf_score)
            in_dense = section in dense_by_section
            in_bm25 = section in bm25_by_section
            if in_dense and in_bm25:
                meta["match_type"] = "hybrid"
            elif in_bm25:
                meta["match_type"] = "bm25"
            else:
                meta["match_type"] = meta.get("match_type") or "dense"
            merged.append(Document(page_content=doc.page_content, metadata=meta))
            seen.add(section)
            if len(merged) >= k:
                break

        return merged[:k]

    def ask_question(self, question):
        """Optimized main method with faster iteration logic"""
        if not self.vectorstore:
            raise RuntimeError("Assistant not initialized. Call initialize() first.")

        # Step 1: Comprehensive initial search (more docs upfront)
        retrieved_docs = self.smart_search_code(question, k=12)

        # Step 2: Fallback with self-query if available
        if len(retrieved_docs) < 5 and self.use_self_query:
            try:
                fallback_docs = self.retriever.invoke(question)[:8]  # Limit fallback docs
                seen_sections = {doc.metadata.get('section') for doc in retrieved_docs}
                for doc in fallback_docs:
                    if doc.metadata.get('section') not in seen_sections:
                        retrieved_docs.append(doc)
                        if len(retrieved_docs) >= 12:  # Cap total docs
                            break
            except Exception:
                pass

        if not retrieved_docs:
            return {
                'success': False,
                'error': 'No relevant documents found',
                'documents': [],
                'answer': None,
                'iterations': 0
            }

        # Step 3: Generate initial response
        try:
            context = self._format_retrieved_docs(retrieved_docs)
            response = self.summary_chain.invoke({
                "question": question,
                "context": context
            })

            answer = response.content if hasattr(response, 'content') else str(response)

            # Step 4: Quick check if we need more info (simplified logic)
            needs_more_info = self._quick_needs_check(answer)

            if not needs_more_info:
                return {
                    'success': True,
                    'documents': retrieved_docs,
                    'answer': answer,
                    'error': None,
                    'iterations': 1
                }

            # Step 5: One additional targeted search if needed
            additional_terms = self._extract_quick_search_terms(answer, question)
            if additional_terms:
                extra_docs = self.batch_search(additional_terms[:3], k_per_query=2)

                # Add unique documents
                seen_sections = {doc.metadata.get('section') for doc in retrieved_docs}
                new_docs = []
                for doc in extra_docs:
                    if doc.metadata.get('section') not in seen_sections:
                        new_docs.append(doc)
                        seen_sections.add(doc.metadata.get('section'))

                if new_docs:
                    all_docs = retrieved_docs + new_docs
                    context = self._format_retrieved_docs(all_docs)

                    # Generate final answer
                    final_response = self.summary_chain.invoke({
                        "question": question,
                        "context": context
                    })

                    final_answer = final_response.content if hasattr(final_response, 'content') else str(final_response)

                    return {
                        'success': True,
                        'documents': all_docs,
                        'answer': final_answer,
                        'error': None,
                        'iterations': 2
                    }

            # Return original answer if no improvement found
            return {
                'success': True,
                'documents': retrieved_docs,
                'answer': answer,
                'error': None,
                'iterations': 1,
                'note': 'Comprehensive search completed'
            }

        except Exception as e:
            return {
                'success': False,
                'documents': retrieved_docs,
                'answer': None,
                'error': f'Error generating response: {e}',
                'iterations': 1
            }

    def _format_retrieved_docs(self, docs):
        """Optimized document formatting"""
        context_parts = []
        seen_sections = set()

        # Limit to top 8 most relevant documents to keep context manageable
        for doc in docs[:8]:
            section = doc.metadata.get('section', 'Unknown')
            if section not in seen_sections:
                # Truncate very long sections to keep context focused
                content = doc.page_content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                context_parts.append(f"Section {section}:\n{content}")
                seen_sections.add(section)

        return "\n\n".join(context_parts)

    def _quick_needs_check(self, answer):
        """Simplified check for whether more information is needed"""
        # Reduced set of indicators for faster processing
        needs_more_indicators = [
            "would need to see",
            "need additional sections",
            "not provided in these sections",
            "additional information"
        ]

        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in needs_more_indicators)
    
    def _extract_quick_search_terms(self, answer, original_question):
        """Fast extraction of search terms without complex AI processing"""
        search_terms = []
        
        # Extract section numbers mentioned
        section_matches = re.findall(r'\b(\d+\.\d+\.\d+(?:\.\d+)*)\b', answer)
        search_terms.extend(section_matches[:2])  # Limit to 2 sections
        
        # Topic-based quick mapping
        question_lower = original_question.lower()
        
        if 'fence' in question_lower or 'wall' in question_lower:
            search_terms.append("fence height zoning")
        elif 'animal' in question_lower:
            search_terms.append("animal control ordinance")
        elif 'business' in question_lower:
            search_terms.append("business license")
        elif 'permit' in question_lower:
            search_terms.append("permit requirements")
        
        return search_terms[:3]  # Limit to 3 terms max
    
    def _check_if_needs_more_sections(self, answer):
        """Check if the AI response indicates it needs more information"""
        needs_more_indicators = [
            "would need to see",
            "need to see more sections",
            "these specific sections do not provide",
            "does not contain any information about",
            "would need additional sections",
            "I would need to see other sections",
            "these sections don't address",
            "need more information",
            "require additional sections",
            "not provided in these sections",
            "would need access to",
            "additional ordinances",
            "other parts of the code",
            "need to consult other sections"
        ]
        
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in needs_more_indicators)
    
    def _extract_section_numbers(self, answer):
        """Extract specific section numbers mentioned in the answer"""
        # Match patterns like "Section 10.16.090" or "10.16.090" or "Section X.Y.Z"
        section_patterns = [
            r'Section\s+(\d+\.\d+\.\d+(?:\.\d+)*)',
            r'\b(\d+\.\d+\.\d+(?:\.\d+)*)\b',
            r'Chapter\s+(\d+\.\d+)'
        ]
        
        sections = []
        for pattern in section_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            sections.extend(matches)
        
        return list(set(sections))  # Remove duplicates
    
    def _get_topic_based_searches(self, original_question, answer):
        """Get search terms based on question topic and common legal areas"""
        question_lower = original_question.lower()
        answer_lower = answer.lower()
        searches = []
        
        # Topic-specific search mappings
        topic_mappings = {
            # Public behavior/decency
            ('shit', 'defecate', 'urinate', 'pee', 'bathroom', 'toilet'): [
                "public decency ordinance",
                "disorderly conduct public",
                "nuisance public behavior",
                "sanitation violations",
                "public restroom requirements"
            ],
            # Animals
            ('tiger', 'lion', 'bear', 'wolf', 'exotic animal', 'wild animal'): [
                "dangerous animal ownership",
                "exotic animal permits",
                "wild animal prohibition",
                "animal control regulations"
            ],
            # Property/construction
            ('fence', 'wall', 'build', 'construct', 'height'): [
                "fence height regulations",
                "residential building codes",
                "property line restrictions",
                "zoning setback requirements"
            ],
            # Business/commercial
            ('business', 'store', 'commercial', 'license'): [
                "business license requirements",
                "commercial zoning regulations",
                "permit commercial activity"
            ],
            # Noise/disturbance
            ('noise', 'loud', 'music', 'party'): [
                "noise ordinance",
                "disturbance public peace",
                "quiet hours regulations"
            ]
        }
        
        # Find matching topics
        for keywords, search_terms in topic_mappings.items():
            if any(keyword in question_lower for keyword in keywords):
                searches.extend(search_terms)
                break
        
        # Context-based additions from the answer
        if "public decency" in answer_lower:
            searches.extend(["public indecency definition", "offenses against decency"])
        if "permit" in answer_lower and "required" in answer_lower:
            searches.append("permit application requirements")
        if "zoning" in answer_lower:
            searches.append("zoning code regulations")
        if "prohibited" not in answer_lower and "allowed" not in answer_lower:
            searches.append("prohibited activities ordinance")
        
        return searches[:4]  # Limit to 4 search terms
    
    def _extract_additional_search_terms(self, answer, original_question):
        """Extract search terms for additional information needed"""
        additional_queries = []
        
        # 1. First priority: Extract specific section numbers mentioned
        section_numbers = self._extract_section_numbers(answer)
        additional_queries.extend(section_numbers)
        
        # 2. Second priority: Get topic-based searches
        topic_searches = self._get_topic_based_searches(original_question, answer)
        additional_queries.extend(topic_searches)
        
        # 3. Third priority: Try AI-based extraction (but don't rely on it)
        try:
            extraction_prompt = PromptTemplate(
                input_variables=["question", "current_answer"],
                template="""Based on this question and partial answer, what specific municipal code topics should I search for?

QUESTION: {question}
CURRENT ANSWER: {current_answer}

Provide 2 short search terms (3-4 words each) for missing information:
SEARCH: [term 1]
SEARCH: [term 2]"""
            )
            
            extraction_chain = extraction_prompt | self.llm
            response = extraction_chain.invoke({
                "question": original_question,
                "current_answer": answer
            })
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract search terms from the response
            search_terms = re.findall(r'SEARCH:\s*(.+)', content, re.IGNORECASE)
            ai_terms = [term.strip() for term in search_terms if term.strip()]
            additional_queries.extend(ai_terms)
            
        except Exception:
            pass  # Don't worry if AI extraction fails
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in additional_queries:
            if query.lower() not in seen:
                seen.add(query.lower())
                unique_queries.append(query)
        
        return unique_queries[:5]  # Limit to 5 total search terms
    
    def search_code(self, question, k=8):
        """Legacy method - redirects to optimized version for compatibility"""
        return self.smart_search_code(question, k)