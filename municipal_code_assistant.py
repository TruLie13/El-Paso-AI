import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate


class MunicipalCodeAssistant:
    """Core assistant for querying El Paso municipal code"""
    
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
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
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
        """Create a chain for summarizing search results"""
        summary_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an expert on El Paso municipal code. Based on the following municipal code sections, provide a clear, helpful answer to the user's question.

QUESTION: {question}

RELEVANT CODE SECTIONS:
{context}

INSTRUCTIONS:
- Provide a direct, practical answer to the question
- Include specific section numbers when referencing rules
- If there are height limits, measurements, or specific requirements, state them clearly
- If the answer depends on zoning or property type, mention that
- Keep the response conversational and helpful
- If you need to see more sections or the question is unclear, say so

ANSWER:"""
        )
        return summary_prompt | self.llm
    
    def search_code(self, question, k=8):
        """Perform enhanced search with multiple query variations"""
        search_variations = [question]
        
        # Add specific variations based on question content
        if any(word in question.lower() for word in ['fence', 'wall', 'height', 'high', 'tall']):
            search_variations.extend([
                "residential fence height limits zoning",
                "permitted walls fence residential apartment",
                "screening wall fence height residential",
                "20.16.030 fence height",
                "fence height residential zoning limits"
            ])
        
        if any(word in question.lower() for word in ['permit', 'build', 'construct', 'install']):
            search_variations.extend([
                "building permit fence wall",
                "fence permit requirements"
            ])
        
        all_docs = []
        seen_content = set()
        
        for variation in search_variations:
            try:
                docs = self.vectorstore.similarity_search(variation, k=k//len(search_variations) + 1)
                for doc in docs:
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(content_hash)
            except Exception:
                continue
        
        # Score and sort results
        question_words = set(question.lower().split())
        
        def relevance_score(doc):
            content_words = set(doc.page_content.lower().split())
            section = doc.metadata.get('section', '')
            
            word_match_score = len(question_words.intersection(content_words))
            
            section_bonus = 0
            if section.startswith('20.16'):
                section_bonus = 10
            elif section.startswith('18.'):
                section_bonus = 5
                
            return word_match_score + section_bonus
        
        all_docs.sort(key=relevance_score, reverse=True)
        return all_docs[:k]
    
    def ask_question(self, question):
        """Main method to ask a question and get results"""
        if not self.vectorstore:
            raise RuntimeError("Assistant not initialized. Call initialize() first.")
        
        # Search for relevant documents
        retrieved_docs = self.search_code(question, k=6)
        
        # Fallback search if not enough results
        if len(retrieved_docs) < 3 and self.use_self_query:
            try:
                fallback_docs = self.retriever.invoke(question)
                seen_sections = {doc.metadata.get('section') for doc in retrieved_docs}
                for doc in fallback_docs:
                    if doc.metadata.get('section') not in seen_sections:
                        retrieved_docs.append(doc)
            except Exception:
                pass
        
        if not retrieved_docs:
            return {
                'success': False,
                'error': 'No relevant documents found',
                'documents': [],
                'answer': None
            }
        
        # Generate AI summary
        try:
            context = self._format_retrieved_docs(retrieved_docs)
            response = self.summary_chain.invoke({
                "question": question,
                "context": context
            })
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                'success': True,
                'documents': retrieved_docs,
                'answer': answer,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': True,
                'documents': retrieved_docs,
                'answer': None,
                'error': f'Error generating AI summary: {e}'
            }
    
    def _format_retrieved_docs(self, docs):
        """Format retrieved documents for the AI prompt"""
        context_parts = []
        seen_sections = set()
        
        for doc in docs:
            section = doc.metadata.get('section', 'Unknown')
            if section not in seen_sections:
                content = doc.page_content
                context_parts.append(f"Section {section}:\n{content}")
                seen_sections.add(section)
        
        return "\n\n".join(context_parts)