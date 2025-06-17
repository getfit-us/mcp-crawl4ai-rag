"""Tests for crawl_single_page tool."""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from types import SimpleNamespace

from crawl4ai_mcp.tools.crawl_single_page import crawl_single_page


@pytest.fixture
def mock_context():
    """Create mock MCP context."""
    context = Mock()
    
    # Create lifespan context
    lifespan_context = SimpleNamespace(
        crawler=Mock(),
        supabase_client=Mock(),
        settings=SimpleNamespace(
            default_chunk_size=5000,
            use_contextual_embeddings=False,
            use_agentic_rag=False,
            openai_api_key="test-key",
            enable_batch_summaries=True,
            enable_batch_embeddings=True,
            enable_batch_contextual_embeddings=True,
            summary_batch_size=10,
            embedding_batch_size=100,
            contextual_embedding_batch_size=20,
            disable_thinking=False
        )
    )
    
    context.request_context.lifespan_context = lifespan_context
    return context


@pytest.fixture
def mock_services():
    """Create mock services."""
    with patch('crawl4ai_mcp.tools.crawl_single_page.EmbeddingService') as MockEmbedding, \
         patch('crawl4ai_mcp.tools.crawl_single_page.DatabaseService') as MockDatabase, \
         patch('crawl4ai_mcp.tools.crawl_single_page.CrawlingService') as MockCrawling, \
         patch('crawl4ai_mcp.tools.crawl_single_page.TextProcessor') as MockTextProcessor:
        
        # Mock embedding service
        embedding_instance = Mock()
        embedding_instance.create_embedding = AsyncMock(return_value=[0.1] * 1536)
        # Add batch processing methods
        embedding_instance.create_embeddings_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536])
        embedding_instance.generate_contextual_embeddings_batch = AsyncMock(return_value=[
            ("Contextual chunk 1", True),
            ("Contextual chunk 2", True),
            ("Contextual chunk 3", True)
        ])
        MockEmbedding.return_value = embedding_instance
        
        # Mock database service
        database_instance = Mock()
        database_instance.update_source_info = AsyncMock(return_value={"success": True})
        database_instance.add_documents = AsyncMock(return_value={"success": True, "count": 3})
        database_instance.add_code_examples = AsyncMock(return_value={"success": True, "count": 2})
        MockDatabase.return_value = database_instance
        
        # Mock crawling service
        crawling_instance = Mock()
        crawling_instance.crawl_batch = AsyncMock(return_value=[
            {"url": "https://example.com", "markdown": "# Test Content\n\nSome content here."}
        ])
        crawling_instance.extract_source_summary = AsyncMock(return_value="Test source summary")
        crawling_instance.extract_code_blocks = Mock(return_value=[])
        crawling_instance.generate_code_example_summary = AsyncMock(return_value="Code summary")
        
        # Add batch processing methods
        crawling_instance.extract_source_summaries_batch = AsyncMock(return_value=["Source summary 1"])
        crawling_instance.generate_code_example_summaries_batch = AsyncMock(return_value=["Code summary 1", "Code summary 2"])
        
        # Mock the new cancellation methods
        crawling_instance.start_crawl_operation = AsyncMock(return_value="test-crawl-id-456")
        crawling_instance.finish_crawl_operation = AsyncMock(return_value=None)
        crawling_instance.cancel_crawl_operation = AsyncMock(return_value=True)
        crawling_instance.cancel_all_crawl_operations = AsyncMock(return_value=0)
        crawling_instance.get_active_crawl_operations = AsyncMock(return_value={})
        
        MockCrawling.return_value = crawling_instance
        
        # Mock text processor
        text_processor_instance = Mock()
        text_processor_instance.smart_chunk_markdown = Mock(return_value=[
            "Chunk 1", "Chunk 2", "Chunk 3"
        ])
        text_processor_instance.extract_section_info = Mock(return_value={
            "headers": "# Test",
            "char_count": 100,
            "word_count": 20
        })
        text_processor_instance.generate_contextual_embedding = AsyncMock(
            return_value=("Contextual chunk", True)
        )
        MockTextProcessor.return_value = text_processor_instance
        
        yield {
            'embedding': embedding_instance,
            'database': database_instance,
            'crawling': crawling_instance,
            'text_processor': text_processor_instance
        }


@pytest.mark.asyncio
async def test_crawl_single_page_success(mock_context, mock_services) -> None:
    """Test successful single page crawl."""
    result = await crawl_single_page(mock_context, "https://example.com")
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    assert result_data["url"] == "https://example.com"
    assert result_data["chunks_created"] == 3
    assert result_data["code_examples_created"] == 0
    assert result_data["source_id"] == "example.com"
    assert "Successfully crawled" in result_data["message"]
    
    # Verify services were called
    mock_services['crawling'].crawl_batch.assert_called_once_with(
        ["https://example.com"], max_concurrent=1, crawl_id="test-crawl-id-456"
    )
    mock_services['text_processor'].smart_chunk_markdown.assert_called_once()
    mock_services['database'].update_source_info.assert_called_once()
    mock_services['database'].add_documents.assert_called_once()


@pytest.mark.asyncio
async def test_crawl_single_page_with_code_examples(mock_context, mock_services) -> None:
    """Test crawling with code example extraction."""
    # Enable code extraction
    mock_context.request_context.lifespan_context.settings.use_agentic_rag = True
    
    # Mock code blocks
    mock_services['crawling'].extract_code_blocks.return_value = [
        {
            'code': 'def hello(): pass',
            'language': 'python',
            'context_before': 'Before',
            'context_after': 'After'
        },
        {
            'code': 'function test() {}',
            'language': 'javascript',
            'context_before': 'Before JS',
            'context_after': 'After JS'
        }
    ]
    
    result = await crawl_single_page(mock_context, "https://example.com")
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    assert result_data["code_examples_created"] == 2
    
    # Verify code example processing
    # The fix ensures that extract_code_blocks is called with min_length=50 instead of default 1000
    mock_services['crawling'].extract_code_blocks.assert_called_once()
    mock_services['database'].add_code_examples.assert_called_once()
    # With batch processing enabled, should use batch method instead of individual calls
    mock_services['crawling'].generate_code_example_summaries_batch.assert_called_once()


@pytest.mark.asyncio
async def test_crawl_single_page_with_contextual_embeddings(mock_context, mock_services) -> None:
    """Test crawling with contextual embeddings."""
    # Enable contextual embeddings
    mock_context.request_context.lifespan_context.settings.use_contextual_embeddings = True
    
    result = await crawl_single_page(mock_context, "https://example.com")
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    
    # With batch processing enabled, should use batch contextual embeddings instead of individual calls
    mock_services['embedding'].generate_contextual_embeddings_batch.assert_called_once()


@pytest.mark.asyncio
async def test_crawl_single_page_crawl_failure(mock_context, mock_services) -> None:
    """Test handling of crawl failure."""
    # Mock empty results
    mock_services['crawling'].crawl_batch.return_value = []
    
    result = await crawl_single_page(mock_context, "https://example.com")
    result_data = json.loads(result)
    
    assert result_data["success"] is False
    assert "Failed to crawl the URL" in result_data["error"]


@pytest.mark.asyncio
async def test_crawl_single_page_exception_handling(mock_context, mock_services) -> None:
    """Test exception handling."""
    # Mock exception
    mock_services['crawling'].crawl_batch.side_effect = Exception("Network error")
    
    result = await crawl_single_page(mock_context, "https://example.com")
    result_data = json.loads(result)
    
    assert result_data["success"] is False
    assert "Network error" in result_data["error"]
    assert result_data["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_metadata_timestamp_format(mock_context, mock_services) -> None:
    """Test that metadata contains properly formatted timestamp."""
    # Mock successful crawl
    mock_services['crawling'].crawl_batch.return_value = [
        {"url": "https://example.com", "markdown": "Test content"}
    ]
    
    # Track the metadata passed to add_documents
    captured_metadata = []
    async def capture_add_documents(**kwargs):
        captured_metadata.extend(kwargs.get('metadatas', []))
        return {"success": True, "count": len(kwargs.get('contents', []))}
    
    mock_services['database'].add_documents = capture_add_documents
    
    result = await crawl_single_page(mock_context, "https://example.com")
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    assert len(captured_metadata) > 0
    
    # Check each metadata entry
    for metadata in captured_metadata:
        # Verify required fields exist
        assert "crawl_time" in metadata
        assert "crawl_type" in metadata
        assert metadata["crawl_type"] == "single_page"
        assert "source" in metadata
        assert "url" in metadata
        assert "chunk_index" in metadata
        
        # Verify timestamp format (ISO format with timezone)
        crawl_time = metadata["crawl_time"]
        assert isinstance(crawl_time, str)
        assert "T" in crawl_time  # ISO format separator
        assert "+" in crawl_time or "Z" in crawl_time  # Timezone indicator
        
        # Verify it can be parsed as a datetime
        from datetime import datetime
        datetime.fromisoformat(crawl_time.replace('Z', '+00:00'))


@pytest.mark.asyncio
async def test_extract_code_blocks_min_length_fix(mock_context, mock_services) -> None:
    """Test that extract_code_blocks is called with min_length=50 instead of default 1000."""
    # Enable code extraction
    mock_context.request_context.lifespan_context.settings.use_agentic_rag = True
    
    # Use a spy to intercept the actual call
    original_extract = mock_services['crawling'].extract_code_blocks
    
    def extract_spy(*args, **kwargs):
        # Store the arguments for verification
        extract_spy.last_call_args = args
        extract_spy.last_call_kwargs = kwargs
        return []
    
    mock_services['crawling'].extract_code_blocks = extract_spy
    
    result = await crawl_single_page(mock_context, "https://example.com")
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    
    # Verify the call was made with correct parameters
    assert hasattr(extract_spy, 'last_call_args')
    assert len(extract_spy.last_call_args) >= 1
    assert extract_spy.last_call_args[0] == "# Test Content\n\nSome content here."
    
    # Check for min_length parameter (50 instead of default 1000)
    if len(extract_spy.last_call_args) > 1:
        assert extract_spy.last_call_args[1] == 50, f"Expected min_length=50, got {extract_spy.last_call_args[1]}"
    elif 'min_length' in extract_spy.last_call_kwargs:
        assert extract_spy.last_call_kwargs['min_length'] == 50, f"Expected min_length=50, got {extract_spy.last_call_kwargs['min_length']}"
    else:
        assert False, f"min_length parameter not found in call. Args: {extract_spy.last_call_args}, Kwargs: {extract_spy.last_call_kwargs}"