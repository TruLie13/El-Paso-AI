import os
import re
from dotenv import load_dotenv
from datetime import datetime
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate


class MunicipalCodeAssistant:
    """Optimized assistant for querying El Paso municipal code"""
    
    def __init__(self, db_path="chroma_db"):
        self.db_path = db_path
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retriever = None
        self.summary_chain = None
        self.use_self_query = False
        
    def initialize(self):
        """Initialize all AI components"""
        load_dotenv()
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        
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
    
    def _create_summary_chain(self):
        """Create a streamlined chain for summarizing search results"""
        summary_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an expert on El Paso municipal code. Based on these code sections, provide a definitive, practical answer.

QUESTION: {question}

CODE SECTIONS:
{context}

INSTRUCTIONS:
- Give a direct YES/NO answer first
- Cite the specific section that applies
- If you find a relevant section (like public indecency, disorderly conduct, etc.), apply it confidently
- Include penalties/consequences if mentioned in the sections
- Be authoritative - if the law clearly applies, state it definitively
- Only mention "additional information needed" if truly critical information is missing
- Focus on practical guidance

ANSWER:"""
        )
        return summary_prompt | self.llm
    
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
        """Perform multiple searches efficiently without threading"""
        all_docs = []
        seen_content = set()
        
        for query in queries:
            try:
                docs = self.vectorstore.similarity_search(query, k=k_per_query)
                for doc in docs:
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(content_hash)
            except Exception:
                continue
        
        return all_docs
    
    def smart_search_code(self, question, k=10):
        """Optimized search with parallel processing and smarter query selection"""
        # Start with the original question
        search_queries = [question]
        
        # Add strategic variations based on content
        question_lower = question.lower()
        
        # Pre-defined high-value search patterns with more comprehensive coverage
        if any(word in question_lower for word in ['shit', 'defecate', 'urinate', 'pee', 'bathroom', 'toilet', 'public restroom']):
            search_queries.extend([
                "public urination defecation prohibited",
                "indecent conduct public decency",
                "disorderly conduct public behavior",
                "public health sanitation violations",
                "nuisance public place bathroom"
            ])
        
        if any(word in question_lower for word in ['fence', 'wall', 'height']):
            search_queries.extend([
                "residential fence height 20.16.030",
                "fence screening wall residential"
            ])
        
        if any(word in question_lower for word in ['animal', 'tiger', 'pet', 'dog']):
            search_queries.extend([
                "animal control dangerous animals",
                "exotic animal prohibition"
            ])
        
        if any(word in question_lower for word in ['business', 'commercial', 'store']):
            search_queries.extend([
                "business license commercial",
                "zoning commercial activity"
            ])
        
        # Limit total queries to prevent excessive searching
        search_queries = search_queries[:6]  # Increased from 5 to 6 for better coverage
        
        # Perform batch searches
        all_docs = self.batch_search(search_queries, k_per_query=max(2, k//len(search_queries)))
        
        # Enhanced scoring for relevance
        question_words = set(question.lower().split())
        
        def relevance_score(doc):
            content_words = set(doc.page_content.lower().split())
            section = doc.metadata.get('section', '')
            content_lower = doc.page_content.lower()
            
            # Word matching score
            word_match_score = len(question_words.intersection(content_words))
            
            # Section priority scoring - updated for public behavior topics
            section_bonus = 0
            if section.startswith('20.16'):  # Zoning - often relevant
                section_bonus = 15
            elif section.startswith('18.'):   # General regulations
                section_bonus = 10
            elif section.startswith('7.'):    # Animals
                section_bonus = 8
            elif section.startswith('10.'):   # Public safety
                section_bonus = 12
            elif section.startswith('9.'):    # Health/sanitation
                section_bonus = 14
            elif section.startswith('8.'):    # Public conduct/peace
                section_bonus = 16
                
            # Content quality indicators - enhanced for public behavior
            quality_bonus = 0
            if any(word in content_lower for word in ['prohibited', 'unlawful', 'shall not', 'violation']):
                quality_bonus += 8
            if any(word in content_lower for word in ['public place', 'indecent', 'disorderly']):
                quality_bonus += 6
            if any(word in content_lower for word in ['urinate', 'defecate', 'excrete']):
                quality_bonus += 10
            if any(word in content_lower for word in ['permitted', 'allowed', 'shall', 'required']):
                quality_bonus += 3
                
            return word_match_score + section_bonus + quality_bonus
        
        # Sort and return top results
        all_docs.sort(key=relevance_score, reverse=True)
        return all_docs[:k]
    
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