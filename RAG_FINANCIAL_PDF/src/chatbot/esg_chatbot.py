"""Financial ESG chatbot orchestrator integrating RAG, memory, and LLM."""
import re
from typing import Optional, Dict, Any
from loguru import logger
from src.llm.ollama_client import OllamaClient
from src.memory.conversation_memory import ConversationMemory
from src.rag.qdrant_store import QdrantStore


class ESGChatbot:
    """Chatbot orchestrator for Financial ESG analysis with RAG and memory."""
    
    def __init__(self):
        """Initialize the chatbot with all components."""
        self.llm = OllamaClient()
        self.memory = ConversationMemory()
        self.rag_store = QdrantStore()
        
        # System prompt for financial ESG analysis
        self.system_prompt = """You are a professional financial ESG (Environmental, Social, Governance) analyst assistant.

CRITICAL RULES - FOLLOW STRICTLY:
1. ONLY answer questions about ESG (Environmental, Social, Governance) topics and financial reporting
2. ONLY use information from the "Relevant ESG Document Context" section provided below
3. NEVER generate, invent, or assume any data, metrics, or facts not in the provided context
4. READ SOURCES IN ORDER - Source 1 is the most relevant, prioritize it first
5. MATCH COMPANY NAMES - Ensure the source document matches the company mentioned in the query
6. If information is not in the context, clearly state "I don't have that specific information in the available reports"
7. If the question is NOT about ESG or financial topics, politely decline: "I'm an ESG financial analyst. I can only help with questions about environmental, social, and governance reporting. Please ask about ESG metrics, sustainability goals, carbon emissions, or related financial data."
8. Be precise with numbers, dates, companies, and metrics
9. Always cite the source document when providing information
10. Use professional, clear, and objective language
11. For numerical data, present exact figures from the context
12. DOUBLE-CHECK: Before answering, verify the source file name matches the company in the question
13. DO NOT answer questions about: weather, general knowledge, coding, mathematics, personal advice, or any non-ESG topics

RESPONSE STYLE:
- Start with a direct answer to the question
- Support with specific data points from the context
- Organize information clearly with bullet points or structured format
- MANDATORY: ALWAYS end with source citations in format [Source: filename.pdf]
- If multiple companies or years are mentioned, compare them systematically
- IMPORTANT: Always verify you're reading from the correct company's report
- Use only ASCII and standard Unicode characters - avoid special symbols
- NEVER give an answer without citing the source document

TONE: Professional, analytical, fact-based, and helpful"""
        
        logger.info("ESG Chatbot initialized successfully")
    
    def _retrieve_context(self, query: str, limit: int = 20) -> str:
        """
        Retrieve relevant context from RAG store with company-aware filtering.
        
        Args:
            query: User query
            limit: Maximum number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        # Enhanced search for financial queries - get more results for comprehensive answers
        results = self.rag_store.search(
            query=query,
            limit=limit * 2,  # Get more to filter
            score_threshold=0.10  # Even lower threshold for better recall
        )
        
        if not results:
            return ""
        
        # Extract company names from query
        query_lower = query.lower()
        company_keywords = {
            'absa': ['absa'],
            'clicks': ['clicks'],
            'distell': ['distell'],
            'sasol': ['sasol'],
            'pick n pay': ['pick n pay', 'picknpay', 'pick-n-pay'],
            'implats': ['implats', 'impala']
        }
        
        # Identify which company is EXPLICITLY mentioned in the query
        # Only filter if the company name is directly mentioned
        mentioned_company = None
        for company, keywords in company_keywords.items():
            # Check for explicit mention
            for kw in keywords:
                # Simple substring check is sufficient - handles possessives, etc.
                if kw in query_lower:
                    mentioned_company = company
                    break
            if mentioned_company:
                break
        
        # Re-rank results based on company match
        if mentioned_company:
            company_keywords_list = company_keywords[mentioned_company]
            
            # Separate company-matching and non-matching results
            matching_results = []
            
            for result in results:
                metadata = result.get('metadata', {})
                source_file = metadata.get('source_file', '').lower()
                text = result.get('text', '').lower()
                
                # Check if the document is about the mentioned company
                # Prioritize filename match, then check document content
                is_match = any(kw in source_file for kw in company_keywords_list)
                if not is_match:
                    # Check first 1000 characters of text for company mention
                    is_match = any(kw in text[:1000] for kw in company_keywords_list)
                
                if is_match:
                    matching_results.append(result)
            
            # Use only matching results if we have any, otherwise return empty to trigger "no data" response
            if matching_results:
                results = matching_results[:limit]
            else:
                # No company-specific documents found
                results = []
        else:
            # No specific company mentioned, use top results
            results = results[:limit]
        
        # Format context with source information
        context_parts = []
        for idx, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            text = result.get('text', '')
            score = result.get('score', 0)
            
            source_info = f"[Source {idx}]"
            if 'source_file' in metadata:
                source_info += f" {metadata['source_file']}"
            if 'page' in metadata:
                source_info += f" (Page {metadata['page']})"
            
            context_parts.append(f"{source_info}\n{text}\n")
        
        return "\n---\n".join(context_parts)
    
    def _is_esg_related_query(self, query: str) -> bool:
        """
        Check if the query is related to ESG topics.
        
        Args:
            query: User query
            
        Returns:
            True if ESG-related, False otherwise
        """
        query_lower = query.lower()
        
        # ESG-related keywords
        esg_keywords = [
            # Environmental
            'carbon', 'emission', 'climate', 'energy', 'renewable', 'sustainability',
            'environment', 'waste', 'water', 'pollution', 'greenhouse', 'footprint',
            'recycling', 'conservation', 'biodiversity', 'green', 'eco',
            
            # Social
            'social', 'employee', 'diversity', 'inclusion', 'labor', 'human rights',
            'community', 'safety', 'health', 'training', 'workforce', 'equity',
            'gender', 'discrimination', 'welfare',
            
            # Governance
            'governance', 'board', 'ethics', 'compliance', 'transparency', 'audit',
            'risk', 'stakeholder', 'accountability', 'integrity', 'policy',
            
            # ESG general
            'esg', 'sustainability report', 'annual report', 'disclosure',
            'target', 'goal', 'initiative', 'performance', 'metric',
            
            # Company names (implies ESG query about specific company)
            'absa', 'clicks', 'distell', 'sasol', 'pick n pay', 'picknpay', 'implats',
            
            # Financial ESG terms
            'sustainable finance', 'green bond', 'esg score', 'rating',
            'materiality', 'gri', 'tcfd', 'un sdg', 'sdg'
        ]
        
        # Non-ESG keywords that indicate off-topic questions
        off_topic_keywords = [
            'weather', 'temperature today', 'forecast', 'rain',
            'recipe', 'cook', 'food',
            'movie', 'film', 'song', 'music',
            'game', 'sport', 'football', 'soccer',
            'joke', 'funny', 'story',
        ]
        
        # Check for off-topic keywords first
        for keyword in off_topic_keywords:
            if keyword in query_lower:
                return False
        
        # Check for ESG-related keywords
        for keyword in esg_keywords:
            if keyword in query_lower:
                return True
        
        # If query mentions reports, documents, or data (likely ESG-related in this context)
        generic_data_keywords = ['report', 'data', 'information', 'document', 'metric', 'figure', 'number']
        for keyword in generic_data_keywords:
            if keyword in query_lower:
                return True
        
        # Default to True for ambiguous queries to avoid false negatives
        # Let the RAG system handle it if no context is found
        return True
    
    def process_message(
        self,
        message: str,
        session_id: str
    ) -> str:
        """
        Process user message and generate response.
        
        Args:
            message: User message
            session_id: Session identifier
            
        Returns:
            Generated response
        """
        try:
            # Add user message to memory
            self.memory.add_message(session_id, "user", message)
            
            # Retrieve relevant context from RAG
            logger.info(f"Retrieving context for query: {message}")
            context = self._retrieve_context(message)
            
            # Get conversation history
            history = self.memory.get_short_term_context(session_id)
            
            # Check if question is ESG-related before retrieving context
            if not self._is_esg_related_query(message):
                off_topic_response = "I'm an ESG financial analyst specialized in environmental, social, and governance reporting. I can only help with questions about:\n\n" \
                                   "- ESG metrics and sustainability goals\n" \
                                   "- Carbon emissions and climate targets\n" \
                                   "- Energy, water, and waste management\n" \
                                   "- Social responsibility and governance\n" \
                                   "- Financial ESG reporting and disclosures\n\n" \
                                   "Please ask me about ESG-related topics from the available company reports (Absa, Clicks, Distell, Sasol, Pick n Pay)."
                self.memory.add_message(session_id, "assistant", off_topic_response)
                return off_topic_response
            
            # Build prompt with context and history
            if context:
                prompt = f"""Relevant ESG Document Context:
{context}

Based ONLY on the above context, please answer the following question:
{message}

CRITICAL REMINDERS:
- ONLY use information from the provided context above
- DO NOT use any external knowledge or make assumptions
- ALWAYS cite the source file name in your answer using format: [Source: filename.pdf]
- Be specific with data and metrics
- If the context doesn't contain the answer, clearly state: "I don't have that specific information in the available ESG reports."
- MANDATORY: End your response with source citations for all facts mentioned"""
            else:
                prompt = f"""I don't have specific ESG report information that matches your query: "{message}"

This could mean:
- The information isn't in the currently indexed reports
- The query might need to be rephrased
- The specific data point you're asking about wasn't reported

Please try:
- Asking about different companies (Absa, Clicks, Distell, Sasol, Pick n Pay)
- Requesting different metrics (carbon emissions, energy use, water consumption, etc.)
- Specifying a year (reports available for 2021-2023)"""
                
                self.memory.add_message(session_id, "assistant", prompt)
                return prompt
            
            # Generate response using LLM
            logger.info("Generating LLM response")
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.1  # Very low for consistent, factual responses
            )
            
            # Add assistant response to memory
            self.memory.add_message(session_id, "assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = "I apologize, but I encountered an error processing your request. Please try again."
            self.memory.add_message(session_id, "assistant", error_response)
            return error_response
    
    def create_session(self, session_id: str, user_name: Optional[str] = None):
        """Create a new conversation session."""
        self.memory.create_session(session_id, user_name)
        logger.info(f"Created session: {session_id}")
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        self.memory.clear_session(session_id)
        logger.info(f"Cleared session: {session_id}")
    
    def get_session_history(self, session_id: str):
        """Get conversation history for a session."""
        return self.memory.get_recent_history(session_id)
    
    def health_check(self) -> Dict[str, bool]:
        """Check system health."""
        try:
            llm_health = self.llm.health_check()
        except:
            llm_health = False
        
        try:
            qdrant_info = self.rag_store.get_collection_info()
            qdrant_health = bool(qdrant_info)
        except:
            qdrant_health = False
        
        return {
            "llm_api": llm_health,
            "qdrant": qdrant_health,
            "memory": True  # SQLite always available
        }
