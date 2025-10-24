"""RAG Evaluation Metrics for Financial ESG RAG System.

Implements comprehensive metrics including:
- RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- Retrieval metrics (MRR, NDCG, Precision@K, Recall@K)
- Custom ESG-specific metrics
"""
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer, util


class RAGEvaluationMetrics:
    """Comprehensive RAG evaluation metrics."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize evaluation metrics with embedding model."""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Initialized RAG evaluation with model: {embedding_model_name}")
    
    # ===== RAGAS-Style Metrics =====
    
    def faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Measure if the answer is faithful to the retrieved contexts.
        
        Checks if claims in the answer can be verified from the contexts.
        Score: 0.0 (unfaithful) to 1.0 (fully faithful)
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not answer or not contexts:
            return 0.0
        
        # Extract factual claims from answer (sentences with numbers or specific terms)
        answer_sentences = self._split_sentences(answer)
        factual_claims = [s for s in answer_sentences if self._is_factual_claim(s)]
        
        if not factual_claims:
            # No factual claims to verify
            return 1.0
        
        # Join all contexts
        full_context = " ".join(contexts)
        
        # Check how many claims are supported by context
        verified_claims = 0
        for claim in factual_claims:
            if self._is_claim_supported(claim, full_context):
                verified_claims += 1
        
        faithfulness_score = verified_claims / len(factual_claims) if factual_claims else 1.0
        return faithfulness_score
    
    def answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Measure how relevant the answer is to the question.
        
        Uses semantic similarity between question and answer.
        Score: 0.0 (irrelevant) to 1.0 (highly relevant)
        
        Args:
            question: User query
            answer: Generated answer
            
        Returns:
            Answer relevancy score between 0 and 1
        """
        if not question or not answer:
            return 0.0
        
        # Compute embeddings
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        answer_embedding = self.embedding_model.encode(answer, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = util.cos_sim(question_embedding, answer_embedding).item()
        
        # Normalize to 0-1 range (cosine similarity is already -1 to 1, but typically positive)
        relevancy_score = max(0.0, min(1.0, (similarity + 1) / 2))
        
        return relevancy_score
    
    def context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Measure precision of retrieved contexts.
        
        Evaluates if the retrieved contexts are relevant to answering the question.
        Score: 0.0 (irrelevant contexts) to 1.0 (all contexts relevant)
        
        Args:
            question: User query
            contexts: Retrieved context chunks
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Context precision score between 0 and 1
        """
        if not contexts:
            return 0.0
        
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        
        # Calculate relevance of each context
        relevant_contexts = 0
        for context in contexts:
            context_embedding = self.embedding_model.encode(context, convert_to_tensor=True)
            similarity = util.cos_sim(question_embedding, context_embedding).item()
            
            # Threshold for considering a context relevant
            if similarity > 0.3:  # Adjustable threshold
                relevant_contexts += 1
        
        precision = relevant_contexts / len(contexts) if contexts else 0.0
        return precision
    
    def context_recall(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str
    ) -> float:
        """
        Measure recall of retrieved contexts.
        
        Evaluates if all necessary information is in the retrieved contexts.
        Score: 0.0 (missing info) to 1.0 (all necessary info present)
        
        Args:
            question: User query
            contexts: Retrieved context chunks
            ground_truth: Ground truth answer
            
        Returns:
            Context recall score between 0 and 1
        """
        if not ground_truth or not contexts:
            return 0.0
        
        # Extract key facts from ground truth
        ground_truth_sentences = self._split_sentences(ground_truth)
        key_facts = [s for s in ground_truth_sentences if self._is_factual_claim(s)]
        
        if not key_facts:
            return 1.0  # No specific facts to recall
        
        # Join contexts
        full_context = " ".join(contexts)
        
        # Check how many key facts are present in contexts
        recalled_facts = 0
        for fact in key_facts:
            if self._is_claim_supported(fact, full_context):
                recalled_facts += 1
        
        recall = recalled_facts / len(key_facts) if key_facts else 1.0
        return recall
    
    # ===== Retrieval Metrics =====
    
    def mean_reciprocal_rank(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR measures how high the first relevant document appears in results.
        
        Args:
            retrieved_docs: List of retrieved documents with 'id' field
            relevant_doc_ids: List of relevant document IDs
            
        Returns:
            MRR score (0 to 1)
        """
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get('id') or doc.get('metadata', {}).get('source_file', '')
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        return 0.0
    
    def precision_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K.
        
        Measures what fraction of top-K results are relevant.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0 to 1)
        """
        if not retrieved_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in top_k
            if (doc.get('id') or doc.get('metadata', {}).get('source_file', '')) in relevant_doc_ids
        )
        
        return relevant_retrieved / min(k, len(top_k))
    
    def recall_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K.
        
        Measures what fraction of relevant documents are in top-K results.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0 to 1)
        """
        if not relevant_doc_ids:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in top_k
            if (doc.get('id') or doc.get('metadata', {}).get('source_file', '')) in relevant_doc_ids
        )
        
        return relevant_retrieved / len(relevant_doc_ids)
    
    def ndcg_at_k(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevant_doc_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Measures ranking quality with position-based discounting.
        
        Args:
            retrieved_docs: List of retrieved documents with scores
            relevant_doc_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@K score (0 to 1)
        """
        if not retrieved_docs or not relevant_doc_ids:
            return 0.0
        
        top_k = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(top_k, 1):
            doc_id = doc.get('id') or doc.get('metadata', {}).get('source_file', '')
            relevance = 1.0 if doc_id in relevant_doc_ids else 0.0
            dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = [1.0] * min(len(relevant_doc_ids), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    # ===== ESG-Specific Metrics =====
    
    def numerical_accuracy(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """
        Measure accuracy of numerical values in answer.
        
        Critical for ESG reports with specific metrics and targets.
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Numerical accuracy score (0 to 1)
        """
        answer_numbers = self._extract_numbers(answer)
        ground_truth_numbers = self._extract_numbers(ground_truth)
        
        if not ground_truth_numbers:
            return 1.0  # No numbers to verify
        
        if not answer_numbers:
            return 0.0  # Missing all numbers
        
        # Check how many ground truth numbers appear in answer
        matched = 0
        for gt_num in ground_truth_numbers:
            if any(abs(gt_num - ans_num) < 0.01 * abs(gt_num) for ans_num in answer_numbers):
                matched += 1
        
        return matched / len(ground_truth_numbers)
    
    def source_citation_accuracy(
        self,
        answer: str,
        expected_sources: List[str]
    ) -> float:
        """
        Check if answer cites correct sources.
        
        Args:
            answer: Generated answer
            expected_sources: List of expected source file names
            
        Returns:
            Citation accuracy score (0 to 1)
        """
        if not expected_sources:
            return 1.0
        
        # Extract sources mentioned in answer
        answer_lower = answer.lower()
        cited_sources = 0
        
        for source in expected_sources:
            source_name = source.lower().replace('.pdf', '')
            if source_name in answer_lower:
                cited_sources += 1
        
        return cited_sources / len(expected_sources)
    
    def company_accuracy(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Verify the answer discusses the correct company mentioned in question.
        
        Critical for multi-company ESG datasets.
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            Company accuracy score (0 to 1)
        """
        company_keywords = {
            'absa': ['absa'],
            'clicks': ['clicks'],
            'distell': ['distell'],
            'sasol': ['sasol'],
            'pick n pay': ['pick n pay', 'picknpay'],
        }
        
        # Find company mentioned in question
        question_lower = question.lower()
        mentioned_company = None
        for company, keywords in company_keywords.items():
            if any(kw in question_lower for kw in keywords):
                mentioned_company = company
                break
        
        if not mentioned_company:
            return 1.0  # No specific company to verify
        
        # Check if answer and contexts are about the correct company
        answer_lower = answer.lower()
        keywords = company_keywords[mentioned_company]
        
        answer_match = any(kw in answer_lower for kw in keywords)
        contexts_match = any(
            any(kw in ctx.lower() for kw in keywords)
            for ctx in contexts
        )
        
        # Both answer and contexts should mention the correct company
        if answer_match and contexts_match:
            return 1.0
        elif answer_match or contexts_match:
            return 0.5
        else:
            return 0.0
    
    # ===== Helper Methods =====
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Check if sentence contains a factual claim (has numbers or specific terms)."""
        # Check for numbers
        if re.search(r'\d+', sentence):
            return True
        
        # Check for specific ESG terms
        esg_terms = [
            'target', 'goal', 'emissions', 'reduction', 'carbon',
            'energy', 'water', 'waste', 'renewable', 'sustainability'
        ]
        sentence_lower = sentence.lower()
        return any(term in sentence_lower for term in esg_terms)
    
    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by context using semantic similarity."""
        # Simple keyword matching for factual verification
        claim_lower = claim.lower()
        context_lower = context.lower()
        
        # Extract key numbers and terms
        claim_numbers = self._extract_numbers(claim)
        context_numbers = self._extract_numbers(context)
        
        # If claim has numbers, check if they appear in context
        if claim_numbers:
            for c_num in claim_numbers:
                if any(abs(c_num - ctx_num) < 0.01 * abs(c_num) for ctx_num in context_numbers):
                    return True
        
        # Check for significant keyword overlap (at least 40%)
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))
        context_words = set(re.findall(r'\b\w+\b', context_lower))
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        claim_words -= stop_words
        context_words -= stop_words
        
        if not claim_words:
            return False
        
        overlap = len(claim_words & context_words) / len(claim_words)
        return overlap > 0.4
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Find numbers including decimals and percentages
        pattern = r'\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(m) for m in matches]
    
    # ===== Combined Metrics =====
    
    def evaluate_rag_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        expected_sources: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response with all metrics.
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Ground truth answer (optional)
            expected_sources: Expected source documents (optional)
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # RAGAS metrics
        metrics['faithfulness'] = self.faithfulness(question, answer, contexts)
        metrics['answer_relevancy'] = self.answer_relevancy(question, answer)
        metrics['context_precision'] = self.context_precision(question, contexts, ground_truth)
        
        if ground_truth:
            metrics['context_recall'] = self.context_recall(question, contexts, ground_truth)
            metrics['numerical_accuracy'] = self.numerical_accuracy(answer, ground_truth)
        
        # ESG-specific metrics
        metrics['company_accuracy'] = self.company_accuracy(question, answer, contexts)
        
        if expected_sources:
            metrics['source_citation'] = self.source_citation_accuracy(answer, expected_sources)
        
        # Calculate overall score (average of available metrics)
        metrics['overall_score'] = np.mean([v for v in metrics.values() if v is not None])
        
        return metrics
