"""Tool for performing RAG queries."""

import json

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.services.search import SearchService
from crawl4ai_mcp.utilities.reranking import Reranker
from crawl4ai_mcp.models import CrawlContext, SearchRequest, SearchType


@mcp.tool()
async def perform_rag_query(
    ctx: Context,
    query: str,
    source: str = None,
    match_count: int = 5
) -> str:
    """
    Perform a RAG (Retrieval-Augmented Generation) query on stored content.
    
    This tool searches through the stored documents using semantic search and optionally
    keyword search (hybrid search). Results can be filtered by source and are reranked
    for relevance if enabled.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'docs.python.org')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        Search results with relevant content chunks
    """
    try:
        # Get context and services
        context: CrawlContext = ctx.request_context.lifespan_context
        postgres_pool = context.supabase_client  # Field name kept for compatibility
        settings = context.settings
        
        # Initialize search service
        search_service = SearchService(postgres_pool, settings)
        
        # Create search request
        search_request = SearchRequest(
            query=query,
            source=source,
            num_results=match_count,
            semantic_threshold=settings.default_semantic_threshold
        )
        
        # Determine search type
        search_type = SearchType.HYBRID if settings.use_hybrid_search else SearchType.SEMANTIC
        
        # Perform search
        search_response = await search_service.perform_search(
            search_request,
            search_type=search_type,
            include_code_examples=False  # This tool is for documents only
        )
        
        if not search_response.success:
            return json.dumps({
                "success": False,
                "error": search_response.error or "Search failed",
                "results": []
            })
        
        results = search_response.results
        
        # Apply reranking if enabled
        if settings.use_reranking and results:
            # Initialize reranker (will handle custom model paths automatically)
            reranker = Reranker(settings=settings)
            
            # Convert SearchResult objects to dictionaries for reranking
            results_dict = []
            for result in results:
                results_dict.append({
                    "content": result.content,
                    "url": result.url,
                    "source": result.source,
                    "chunk_number": result.chunk_number,
                    "similarity_score": result.similarity_score,
                    "metadata": result.metadata
                })
            
            # Rerank results
            reranked = reranker.rerank_results(query, results_dict)
            
            # Filter by threshold
            reranked = reranker.filter_by_threshold(
                reranked, 
                threshold=settings.default_rerank_threshold
            )
            
            # Use reranked results
            results_dict = reranked
        else:
            # Convert to dictionaries
            results_dict = []
            for result in results:
                results_dict.append({
                    "content": result.content,
                    "url": result.url,
                    "source": result.source,
                    "chunk_number": result.chunk_number,
                    "similarity_score": result.similarity_score,
                    "metadata": result.metadata
                })
        
        # Format results
        formatted_results = []
        for result in results_dict[:match_count]:  # Limit to requested count
            formatted_result = {
                "content": result["content"],
                "url": result["url"],
                "source": result["source"],
                "chunk_number": result["chunk_number"],
                "similarity_score": result["similarity_score"],
                "metadata": result["metadata"]
            }
            
            # Add rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            
            formatted_results.append(formatted_result)
        
        return json.dumps({
            "success": True,
            "query": query,
            "search_type": search_type.value,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "source_filter": source,
            "reranking_applied": settings.use_reranking,
            "message": f"Found {len(formatted_results)} relevant results"
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query,
            "results": []
        })