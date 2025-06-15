"""Search service for document and code example retrieval."""

import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg
from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import SearchRequest, SearchResult, SearchResponse, SearchType
from crawl4ai_mcp.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class SearchService:
    """Service for searching documents and code examples."""
    
    def __init__(self, pool: asyncpg.Pool, settings: Optional[Any] = None, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize search service.
        
        Args:
            pool: AsyncPG connection pool
            settings: Application settings (optional)
            embedding_service: Embedding service instance (optional)
        """
        self.pool = pool
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
    
    async def search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        use_hybrid_search: Optional[bool] = None
    ) -> List[SearchResult]:
        """
        Search for documents using vector similarity.
        
        Args:
            query: Search query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_id: Optional source ID to filter results
            use_hybrid_search: Whether to use hybrid search (overrides settings)
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Create embedding for the query
            query_embedding = await self.embedding_service.create_embedding(query)
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            return []
        
        async with self.pool.acquire() as conn:
            try:
                # Call the PostgreSQL function
                rows = await conn.fetch(
                    "SELECT * FROM match_crawled_pages($1, $2, $3, $4)",
                    query_embedding,
                    match_count,
                    json.dumps(filter_metadata) if filter_metadata else '{}',
                    source_id
                )
                
                # Convert results to SearchResult objects
                search_results = []
                for row in rows:
                    search_results.append(SearchResult(
                        content=row.get('content', ''),
                        url=row.get('url', ''),
                        source=row.get('source_id', ''),
                        chunk_number=row.get('chunk_number', 0),
                        similarity_score=row.get('similarity', 0.0),
                        metadata=row.get('metadata', {})
                    ))
                
                return search_results
                
            except Exception as e:
                logger.error(f"Error searching documents: {e}")
                return []
    
    async def search_code_examples(
        self,
        query: str,
        language: Optional[str] = None,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for code examples using vector similarity.
        
        Args:
            query: Search query text
            language: Optional programming language filter
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_id: Optional source ID to filter results
            
        Returns:
            List of matching code examples
        """
        # Enhance query for better code example matching
        enhanced_query = f"Code example for {query}"
        if language:
            enhanced_query += f" in {language}"
        enhanced_query += f"\n\nSummary: Example code showing {query}"
        
        try:
            # Create embedding for the enhanced query
            query_embedding = await self.embedding_service.create_embedding(enhanced_query)
        except Exception as e:
            logger.error(f"Error creating code example query embedding: {e}")
            return []
        
        # Add language filter if provided
        if language:
            if not filter_metadata:
                filter_metadata = {}
            filter_metadata['language'] = language
        
        async with self.pool.acquire() as conn:
            try:
                # Call the PostgreSQL function
                rows = await conn.fetch(
                    "SELECT * FROM match_code_examples($1, $2, $3, $4)",
                    query_embedding,
                    match_count,
                    json.dumps(filter_metadata) if filter_metadata else '{}',
                    source_id
                )
                
                # Convert rows to dictionaries
                return [dict(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Error searching code examples: {e}")
                return []
    
    async def perform_search(
        self,
        request: SearchRequest,
        search_type: SearchType = SearchType.SEMANTIC,
        include_code_examples: bool = False
    ) -> SearchResponse:
        """
        Perform a search based on request parameters.
        
        Args:
            request: SearchRequest object with search parameters
            search_type: Type of search to perform
            include_code_examples: Whether to include code examples in results
            
        Returns:
            SearchResponse object with results
        """
        try:
            # Search documents
            results = await self.search_documents(
                query=request.query,
                match_count=request.num_results,
                filter_metadata=None,  # Source filtering is handled by source_id parameter
                source_id=request.source,
                use_hybrid_search=(search_type == SearchType.HYBRID)
            )
            
            # Filter by semantic threshold if specified
            if request.semantic_threshold > 0:
                results = [
                    r for r in results 
                    if r.similarity_score >= request.semantic_threshold
                ]
            
            # Search code examples if requested
            code_results = []
            if include_code_examples and self.settings.use_agentic_rag:
                code_data = await self.search_code_examples(
                    query=request.query,
                    match_count=request.num_results,
                    source_id=request.source
                )
                
                # Convert code examples to SearchResult objects
                for code_ex in code_data:
                    code_results.append(SearchResult(
                        content=code_ex.get('content', ''),
                        url=code_ex.get('url', ''),
                        source=code_ex.get('source_id', ''),
                        chunk_number=code_ex.get('chunk_number', 0),
                        similarity_score=code_ex.get('similarity', 0.0),
                        metadata=code_ex.get('metadata', {})
                    ))
            
            # Combine results
            all_results = results + code_results
            
            # Apply reranking if enabled
            if self.settings.use_reranking and hasattr(self, 'reranking_model') and self.reranking_model:
                all_results = await self.rerank_results(
                    query=request.query,
                    results=all_results,
                    reranking_model=self.reranking_model,
                    threshold=request.rerank_threshold or self.settings.default_rerank_threshold
                )
            
            return SearchResponse(
                success=True,
                query=request.query,
                results=all_results,
                total_results=len(all_results),
                search_type=search_type.value
            )
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return SearchResponse(
                success=False,
                query=request.query,
                results=[],
                total_results=0,
                search_type=search_type.value,
                error=str(e)
            )
    
    async def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        reranking_model: Any,
        threshold: float = 0.3
    ) -> List[SearchResult]:
        """
        Rerank search results using a cross-encoder model.
        
        Args:
            query: Original search query
            results: List of search results to rerank
            reranking_model: Cross-encoder model for reranking
            threshold: Minimum score threshold for reranked results
            
        Returns:
            Reranked and filtered list of SearchResult objects
        """
        if not results or not reranking_model:
            return results
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, result.content] for result in results]
            
            # Get reranking scores
            scores = reranking_model.predict(pairs)
            
            # Update results with rerank scores
            for i, (result, score) in enumerate(zip(results, scores)):
                result.rerank_score = float(score)
            
            # Filter by threshold
            reranked_results = [
                r for r in results 
                if r.rerank_score is not None and r.rerank_score >= threshold
            ]
            
            # Sort by rerank score
            reranked_results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original results if reranking fails
            return results