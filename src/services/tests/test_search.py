"""Tests for search service."""

import pytest
from unittest.mock import Mock

from crawl4ai_mcp.services.search import SearchService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.models import SearchRequest, SearchResult


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock(spec=EmbeddingService)
    # Make it an async mock
    async def mock_create_embedding(text):
        return [0.1] * 1536
    service.create_embedding = mock_create_embedding
    return service


@pytest.fixture
def search_service(mock_postgres_pool, test_settings, mock_embedding_service):
    """Create SearchService with mocked dependencies."""
    return SearchService(mock_postgres_pool, test_settings, mock_embedding_service)


@pytest.fixture
def mock_search_results():
    """Mock search results from database."""
    return [
        {
            'content': 'First result content',
            'url': 'https://example.com/1',
            'source_id': 'example.com',  # Fixed: DB returns 'source_id' not 'source'
            'chunk_number': 1,
            'similarity': 0.9,
            'metadata': {'title': 'First'}
        },
        {
            'content': 'Second result content',
            'url': 'https://example.com/2',
            'source_id': 'example.com',  # Fixed: DB returns 'source_id' not 'source'
            'chunk_number': 2,
            'similarity': 0.8,
            'metadata': {'title': 'Second'}
        },
        {
            'content': 'Third result content',
            'url': 'https://example.com/3',
            'source_id': 'example.com',  # Fixed: DB returns 'source_id' not 'source'
            'chunk_number': 3,
            'similarity': 0.7,
            'metadata': {'title': 'Third'}
        }
    ]


@pytest.fixture
def mock_code_results():
    """Mock code example results."""
    return [
        {
            'content': 'def example():\n    return "Hello"',  # Fixed: DB returns 'content' not 'code'
            'url': 'https://example.com/code1',
            'source_id': 'example.com',  # Fixed: DB returns 'source_id' not 'source'
            'chunk_number': 1,
            'similarity': 0.85,
            'summary': 'Example function',
            'metadata': {'language': 'python'}  # Fixed: language is in metadata
        }
    ]


@pytest.mark.asyncio
async def test_search_documents_success(search_service, mock_postgres_pool, mock_search_results, mock_embedding_service) -> None:
    """Test successful document search."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch response
    mock_conn.fetch.return_value = [
        {
            'content': 'First result content',
            'url': 'https://example.com/1',
            'source_id': 'example.com',
            'chunk_number': 1,
            'similarity': 0.9,
            'metadata': {'title': 'First'}
        },
        {
            'content': 'Second result content',
            'url': 'https://example.com/2',
            'source_id': 'example.com',
            'chunk_number': 2,
            'similarity': 0.8,
            'metadata': {'title': 'Second'}
        }
    ]
    
    results = await search_service.search_documents(
        query="test query",
        match_count=10
    )
    
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].content == "First result content"
    assert results[0].similarity_score == 0.9
    assert results[0].url == "https://example.com/1"
    
    # Verify PostgreSQL function was called
    mock_conn.fetch.assert_called_once()
    call_args = mock_conn.fetch.call_args[0]
    assert "match_crawled_pages" in call_args[0]
    assert len(call_args[1]) == 1536  # Embedding vector
    assert call_args[2] == 10  # Match count


@pytest.mark.asyncio
async def test_search_documents_with_filters(search_service, mock_postgres_pool, mock_search_results) -> None:
    """Test document search with metadata filters."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch response
    mock_conn.fetch.return_value = [
        {
            'content': 'Filtered result content',
            'url': 'https://example.com/filtered',
            'source_id': 'docs.example.com',
            'chunk_number': 1,
            'similarity': 0.9,
            'metadata': {'category': 'tutorial'}
        }
    ]
    
    filter_metadata = {'category': 'tutorial'}
    await search_service.search_documents(
        query="test query",
        match_count=5,
        filter_metadata=filter_metadata,
        source_id="docs.example.com"
    )
    
    # Verify filters were applied
    mock_conn.fetch.assert_called_once()
    call_args = mock_conn.fetch.call_args[0]
    assert "match_crawled_pages" in call_args[0]
    assert call_args[3] == filter_metadata  # Filter metadata
    assert call_args[4] == "docs.example.com"  # Source filter


@pytest.mark.asyncio
async def test_search_documents_error_handling(search_service, mock_postgres_pool) -> None:
    """Test error handling in document search."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch to raise an exception
    mock_conn.fetch.side_effect = Exception("Database error")
    
    results = await search_service.search_documents(query="test query")
    
    assert results == []


@pytest.mark.asyncio
async def test_search_documents_embedding_error(search_service, mock_embedding_service) -> None:
    """Test error handling when embedding creation fails."""
    # Mock embedding service to raise an exception
    async def mock_create_embedding_error(text):
        raise Exception("Embedding API error")
    
    mock_embedding_service.create_embedding = mock_create_embedding_error
    
    results = await search_service.search_documents(
        query="test query",
        match_count=10
    )
    
    # Should return empty list when embedding fails
    assert results == []


@pytest.mark.asyncio
async def test_search_code_examples_success(search_service, mock_postgres_pool, mock_code_results) -> None:
    """Test successful code example search."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch response
    mock_conn.fetch.return_value = [
        {
            'content': 'def example():\n    return "Hello"',
            'url': 'https://example.com/code1',
            'source_id': 'example.com',
            'chunk_number': 1,
            'similarity': 0.85,
            'summary': 'Example function',
            'metadata': {'language': 'python'}
        }
    ]
    
    results = await search_service.search_code_examples(
        query="example function",
        language="python",
        match_count=5
    )
    
    assert len(results) == 1
    assert results[0]['content'] == 'def example():\n    return "Hello"'
    assert results[0]['metadata']['language'] == 'python'
    
    # Verify PostgreSQL function was called
    mock_conn.fetch.assert_called_once()
    call_args = mock_conn.fetch.call_args[0]
    assert "match_code_examples" in call_args[0]


@pytest.mark.asyncio
async def test_search_code_examples_with_source_filter(search_service, mock_postgres_pool, mock_code_results) -> None:
    """Test code example search with source filter."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch response
    mock_conn.fetch.return_value = []
    
    await search_service.search_code_examples(
        query="test",
        source_id="github.com"
    )
    
    # Verify source filter was applied
    mock_conn.fetch.assert_called_once()
    call_args = mock_conn.fetch.call_args[0]
    assert "match_code_examples" in call_args[0]
    assert call_args[4] == "github.com"  # Source filter parameter


@pytest.mark.asyncio
async def test_perform_search_documents_only(search_service, mock_postgres_pool, mock_search_results) -> None:
    """Test perform_search with documents only."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch response
    mock_conn.fetch.return_value = [
        {
            'content': 'Test result content',
            'url': 'https://example.com/test',
            'source_id': 'example.com',
            'chunk_number': 1,
            'similarity': 0.9,
            'metadata': {'title': 'Test'}
        }
    ]
    
    request = SearchRequest(
        query="test query",
        num_results=5
    )
    
    response = await search_service.perform_search(request)
    
    assert response.success is True
    assert len(response.results) == 1
    assert response.results[0].content == "Test result content"
    assert response.total_results == 1


@pytest.mark.asyncio
async def test_perform_search_with_code_examples(search_service, mock_postgres_pool, mock_search_results, mock_code_results, test_settings) -> None:
    """Test perform_search including code examples."""
    # Enable code examples
    test_settings.use_agentic_rag = True
    
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch responses for both document and code searches
    mock_conn.fetch.side_effect = [
        [  # Document search results
            {
                'content': 'Document result',
                'url': 'https://example.com/doc',
                'source_id': 'example.com',
                'chunk_number': 1,
                'similarity': 0.9,
                'metadata': {'title': 'Doc'}
            }
        ],
        [  # Code example search results
            {
                'content': 'def test(): pass',
                'url': 'https://example.com/code',
                'source_id': 'example.com',
                'chunk_number': 1,
                'similarity': 0.8,
                'summary': 'Test function',
                'metadata': {'language': 'python'}
            }
        ]
    ]
    
    request = SearchRequest(
        query="test function",
        num_results=5
    )
    
    response = await search_service.perform_search(request, include_code_examples=True)
    
    assert response.success is True
    assert len(response.results) == 2  # Document + code example
    assert response.total_results == 2


@pytest.mark.asyncio
async def test_perform_search_error_handling(search_service, mock_postgres_pool) -> None:
    """Test perform_search error handling."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock fetch to raise an exception
    mock_conn.fetch.side_effect = Exception("Search failed")
    
    request = SearchRequest(query="test query")
    response = await search_service.perform_search(request)
    
    # The search_documents method catches exceptions and returns empty list,
    # so perform_search will return success=True with empty results
    assert response.success is True
    assert response.total_results == 0
    assert len(response.results) == 0


@pytest.mark.asyncio
async def test_rerank_results(search_service) -> None:
    """Test result reranking."""
    # Create mock reranking model
    mock_reranker = Mock()
    mock_reranker.predict.return_value = [0.9, 0.2, 0.6]  # Rerank scores
    
    # Create test results
    results = [
        SearchResult(
            content="First content",
            url="url1",
            source="source1",
            chunk_number=1,
            similarity_score=0.7
        ),
        SearchResult(
            content="Second content",
            url="url2",
            source="source2",
            chunk_number=2,
            similarity_score=0.8
        ),
        SearchResult(
            content="Third content",
            url="url3",
            source="source3",
            chunk_number=3,
            similarity_score=0.6
        )
    ]
    
    reranked = await search_service.rerank_results(
        query="test query",
        results=results,
        reranking_model=mock_reranker,
        threshold=0.5
    )
    
    # Only results with rerank score >= 0.5 should be included
    assert len(reranked) == 2
    assert reranked[0].rerank_score == 0.9  # First result
    assert reranked[1].rerank_score == 0.6  # Third result
    
    # Verify reranker was called correctly
    expected_pairs = [
        ["test query", "First content"],
        ["test query", "Second content"],
        ["test query", "Third content"]
    ]
    mock_reranker.predict.assert_called_once_with(expected_pairs)


@pytest.mark.asyncio
async def test_rerank_results_empty_input(search_service) -> None:
    """Test reranking with empty results."""
    mock_reranker = Mock()
    
    # Test empty results
    results = await search_service.rerank_results(
        query="test",
        results=[],
        reranking_model=mock_reranker
    )
    assert results == []
    
    # Test None reranker
    results = await search_service.rerank_results(
        query="test",
        results=[SearchResult(content="test", url="url", source="src", chunk_number=1, similarity_score=0.5)],
        reranking_model=None
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_rerank_results_error_handling(search_service) -> None:
    """Test reranking error handling."""
    mock_reranker = Mock()
    mock_reranker.predict.side_effect = Exception("Reranking failed")
    
    original_results = [
        SearchResult(content="test", url="url", source="src", chunk_number=1, similarity_score=0.5)
    ]
    
    results = await search_service.rerank_results(
        query="test",
        results=original_results,
        reranking_model=mock_reranker
    )
    
    # Should return original results on error
    assert results == original_results