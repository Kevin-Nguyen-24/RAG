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
    
    def _retrieve_context(self, query: str, limit: int = 10) -> str:
        """
        Retrieve relevant context from RAG store with company-aware filtering.
        
        Args:
            query: User query
            limit: Maximum number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        # Enhanced search for financial queries - balanced retrieval
        results = self.rag_store.search(
            query=query,
            limit=limit,  # Use limit directly to avoid context overload
            score_threshold=0.15  # Lower threshold to improve recall of numeric facts
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
        max_chars_per_doc = 800  # Limit each document chunk to 800 chars
        
        for idx, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            text = result.get('text', '')
            score = result.get('score', 0)
            
            # Truncate text if too long
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc] + "...[truncated]"
            
            source_info = f"[Source {idx}]"
            if 'source_file' in metadata:
                source_info += f" {metadata['source_file']}"
            if 'page' in metadata:
                source_info += f" (Page {metadata['page']})"
            
            context_parts.append(f"{source_info}\n{text}\n")
        
        # Final safety check - limit total context size
        full_context = "\n---\n".join(context_parts)
        max_total_context = 10000  # ~10KB limit for entire context
        
        if len(full_context) > max_total_context:
            logger.warning(f"Context too large ({len(full_context)} chars), truncating to {max_total_context}")
            full_context = full_context[:max_total_context] + "\n\n[Context truncated due to length]"
        
        return full_context
    
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
    
    def _looks_numeric_query(self, query: str) -> bool:
        """Heuristic: determine if the query asks for numeric ESG facts."""
        q = query.lower()
        numeric_keywords = [
            'kwh', 'kw h', 'tonne', 'tonnes', 't co', 'tco2', 'co₂', 'co2', 'scope 1', 'scope1', 'scope-1',
            'emission', 'energy', 'electricity', 'reduction', 'target', 'percent', '%', '2021', '2022', '2030', 'difference', 'change'
        ]
        return any(k in q for k in numeric_keywords)

    def _parse_context_sources(self, context: str):
        """Split the formatted context into list of {file, page, text}."""
        segments = []
        for block in context.split('\n---\n'):
            block = block.strip()
            if not block:
                continue
            # Header like: [Source 1] filename.pdf (Page 12)
            m = re.match(r"\[Source\s+\d+\]\s*(?P<file>[^\n\(]+)?(?:\s*\(Page\s*(?P<page>\d+)\))?\s*\n(?P<text>[\s\S]*)", block, flags=re.IGNORECASE)
            if m:
                segments.append({
                    'file': (m.group('file') or '').strip(),
                    'page': m.group('page'),
                    'text': (m.group('text') or '').strip(),
                })
            else:
                segments.append({'file': '', 'page': None, 'text': block})
        return segments

    def _num(self, s: str) -> Optional[float]:
        try:
            # normalize 215,963,015 -> 215963015
            return float(re.sub(r'[^0-9\.-]', '', s))
        except Exception:
            return None

    def _try_answer_structured_numeric(self, query: str, context: str) -> Optional[str]:
        """Attempt to answer numeric questions deterministically from context."""
        q = query.lower()
        segs = self._parse_context_sources(context)

        def cite(file):
            return f"[Source: {file}]" if file else "[Source: ESG report]"

        # Q1: total energy use (kWh) 2022
        if ('total' in q and 'energy' in q and '2022' in q) or ('energy use' in q and '2022' in q):
            best = None
            best_file = ''
            for s in segs:
                for m in re.finditer(r"total\s+energy[^\n]*?(\d[\d,\.]+)\s*(kwh|mwh)", s['text'], flags=re.I):
                    val, unit = m.group(1), m.group(2)
                    # Prefer kWh; if MWh, convert to kWh
                    n = self._num(val)
                    if n is None:
                        continue
                    if unit.lower() == 'mwh':
                        n = n * 1000.0
                    if not best or n > best:
                        best = n
                        best_file = s['file']
            if best:
                formatted = f"{int(best):,} kWh"
                return f"{formatted} {cite(best_file)}"

        # Q2: total carbon emissions (tonnes) 2022
        if ('total' in q and 'emission' in q and '2022' in q) and ('co2' in q or 'co₂' in q or 'carbon' in q):
            best = None
            best_file = ''
            for s in segs:
                for m in re.finditer(r"total[^\n]*?emissions[^\n]*?(\d[\d,\.]+)\s*(t|tonnes|tons)\s*(co2|co₂|co2e)?", s['text'], flags=re.I):
                    n = self._num(m.group(1))
                    if n is None:
                        continue
                    if not best or n > best:
                        best = n; best_file = s['file']
            if best:
                return f"{int(best):,} tonnes CO₂ {cite(best_file)}"

        # Q3: 2030 reduction target
        if ('2030' in q and ('reduction' in q or 'target' in q) and ('carbon' in q or 'emission' in q)):
            for s in segs:
                m = re.search(r"(\d{1,3})\s*%[^\n]*?(reduction|reduce)[^\n]*?2030", s['text'], flags=re.I)
                if m:
                    pct = m.group(1)
                    return f"A {pct} % reduction in carbon emissions by 2030. {cite(s['file'])}"

        # Q4: 2022 Scope 1 emissions and components
        if ('scope 1' in q and ('component' in q or 'gas' in q or 'diesel' in q or 'company car' in q)):
            total = gas = cars = diesel = None
            src_file = ''
            for s in segs:
                text = s['text']
                if re.search(r"scope\s*1", text, flags=re.I):
                    if not src_file:
                        src_file = s['file']
                    m_tot = re.search(r"scope\s*1[^\n]*?total[^\n]*?(\d[\d,\.]+)\s*(t|tonnes)", text, flags=re.I)
                    if m_tot:
                        total = self._num(m_tot.group(1))
                    m_g = re.search(r"\bgas\b[^\n]*?(\d[\d,\.]+)\s*(t|tonnes)", text, flags=re.I)
                    if m_g:
                        gas = self._num(m_g.group(1))
                    m_c = re.search(r"company\s*cars?[^\n]*?(\d[\d,\.]+)\s*(t|tonnes)", text, flags=re.I)
                    if m_c:
                        cars = self._num(m_c.group(1))
                    m_d = re.search(r"diesel[^\n]*?(\d[\d,\.]+)\s*(t|tonnes)", text, flags=re.I)
                    if m_d:
                        diesel = self._num(m_d.group(1))
            if total and gas and cars and diesel:
                return (
                    f"Total = {int(total):,} t CO₂; Gas {int(gas):,} t, Company cars {int(cars):,} t, Diesel {int(diesel):,} t. "
                    f"{cite(src_file)}"
                )

        # Q5: Non-renewable electricity change 2021→2022
        if ('non-renewable' in q and 'electricity' in q and ('change' in q or '2021' in q and '2022' in q)):
            y2021 = y2022 = None
            src_file = ''
            for s in segs:
                text = s['text']
                # Try to capture rows like: Non-renewable electricity ... 2021 180,036,673 ... 2022 157,530,491
                for m in re.finditer(r"non[-\s]?renewable\s+electricity[^\n]*", text, flags=re.I):
                    line = m.group(0)
                    m21 = re.search(r"2021[^\d]*(\d[\d,\.]+)", line)
                    m22 = re.search(r"2022[^\d]*(\d[\d,\.]+)", line)
                    if m21:
                        y2021 = self._num(m21.group(1))
                    if m22:
                        y2022 = self._num(m22.group(1))
                    if (y2021 or y2022) and not src_file:
                        src_file = s['file']
                # Fallback: search nearby lines
                if y2021 is None:
                    m21b = re.search(r"non[-\s]?renewable\s+electricity[^\n\r]*\n[^\n\r]*2021[^\d]*(\d[\d,\.]+)", text, flags=re.I)
                    if m21b:
                        y2021 = self._num(m21b.group(1))
                        if not src_file: src_file = s['file']
                if y2022 is None:
                    m22b = re.search(r"non[-\s]?renewable\s+electricity[^\n\r]*\n[^\n\r]*2022[^\d]*(\d[\d,\.]+)", text, flags=re.I)
                    if m22b:
                        y2022 = self._num(m22b.group(1))
                        if not src_file: src_file = s['file']
            if y2021 is not None and y2022 is not None:
                diff = y2022 - y2021
                sign = 'Decreased' if diff < 0 else 'Increased'
                return f"{sign} by {abs(int(diff)):,} kWh ({int(y2021):,} → {int(y2022):,}). {cite(src_file)}"

        return None

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
            
            # If numeric fact-style query, try deterministic extraction first
            if context and self._looks_numeric_query(message):
                numeric_answer = self._try_answer_structured_numeric(message, context)
                if numeric_answer:
                    self.memory.add_message(session_id, "assistant", numeric_answer)
                    return numeric_answer

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
- Be specific with data and metrics (prefer exact figures and units)
- If you need to compute a change between years, show the numbers and the difference succinctly
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
                temperature=0.0  # Deterministic to minimize hallucinations
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
