"""Qdrant vector store for ESG financial document embeddings."""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from pydantic import ValidationError
import shutil
import re
from loguru import logger
from src.config import config


class QdrantStore:
    """Manages Qdrant vector database for financial ESG RAG."""
    
    def __init__(self):
        """Initialize Qdrant client and embedding model."""
        # Try server first, then use persistent file-based storage
        try:
            # First try to connect to server if available
            self.client = QdrantClient(
                host=config.qdrant.host,
                port=config.qdrant.port
            )
            # Test connection
            self.client.get_collections()
            logger.info(f"Connected to Qdrant server at {config.qdrant.host}:{config.qdrant.port}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant server: {e}")
            # Use file-based storage for persistence
            from pathlib import Path
            qdrant_path = Path("./data/qdrant_db")
            qdrant_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local file-based Qdrant storage: {qdrant_path}")
            try:
                self.client = QdrantClient(path=str(qdrant_path))
            except ValidationError as ve:
                logger.error(f"Validation error loading local Qdrant data: {ve}. Re-initializing...")
                shutil.rmtree(qdrant_path)
                qdrant_path.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(qdrant_path))
        
        self.collection_name = config.qdrant.collection_name
        self.embedding_model = SentenceTransformer(config.embedding.model_name)
        self.vector_size = config.qdrant.vector_size
        
        self._ensure_collection_exists()
        logger.info(f"Initialized Qdrant store: {self.collection_name}")
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.debug(f"Collection exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                documents,
                batch_size=config.embedding.batch_size,
                show_progress_bar=True
            )
            
            # Prepare points
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point_id = ids[idx] if ids else idx
                payload = {
                    "text": doc,
                    **(metadata[idx] if metadata and idx < len(metadata) else {})
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(documents)} documents to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.3,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents with enhanced financial data matching.
        Handles both textual queries and numerical data queries well.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional metadata filters
            
        Returns:
            List of search results with text and metadata
        """
        try:
            # Enhanced query processing for financial data
            search_queries = [query]
            
            # Detect if query contains numbers (financial metrics)
            has_numbers = bool(re.search(r'\d+', query))
            query_lower = query.lower()
            
            # Financial metric keywords
            financial_keywords = [
                'carbon', 'emissions', 'energy', 'water', 'waste',
                'target', 'reduction', 'sustainability', 'esg',
                'revenue', 'profit', 'investment', 'cost',
                'kwh', 'tonnes', 'tons', 'percent', '%'
            ]
            
            # Company names from ESG reports
            company_keywords = ['absa', 'clicks', 'distell', 'sasol', 'pick n pay', 'picknpay']
            
            # Extract year if mentioned
            year_match = re.search(r'20\d{2}', query)
            
            # Add enhanced query variations
            if has_numbers or any(kw in query_lower for kw in financial_keywords):
                # For numerical queries, add context-rich variations
                search_queries.append(f"financial data metrics {query}")
                search_queries.append(f"esg report {query}")
                
            # Add company-specific queries
            for company in company_keywords:
                if company in query_lower:
                    search_queries.append(f"{company} sustainability report {query}")
                    if year_match:
                        search_queries.append(f"{company} {year_match.group()} esg")
            
            # Collect all results from different query variations
            all_results = []
            seen_ids = set()
            
            for search_query in search_queries:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(search_query)
                
                # Prepare filter
                search_filter = None
                if filter_conditions:
                    conditions = [
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                    search_filter = Filter(must=conditions)
                
                # Search Qdrant
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=limit * 2,  # Get more results to filter
                    score_threshold=max(score_threshold - 0.1, 0.2),  # Slightly lower threshold
                    query_filter=search_filter
                )
                
                # Add unique results
                for result in search_results:
                    result_id = str(result.id)
                    if result_id not in seen_ids:
                        seen_ids.add(result_id)
                        all_results.append({
                            "text": result.payload.get("text", ""),
                            "score": result.score,
                            "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                        })
            
            # Sort by score (highest first) and limit results
            all_results.sort(key=lambda x: x['score'], reverse=True)
            final_results = all_results[:limit]
            
            logger.debug(f"Found {len(final_results)} relevant documents")
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information dictionary
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def close(self):
        """Close the Qdrant client and release resources."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.debug("Closed Qdrant client")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")
