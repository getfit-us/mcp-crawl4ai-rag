"""Tests for embeddings service."""

import pytest
from unittest.mock import Mock, patch
import openai
from types import SimpleNamespace

from crawl4ai_mcp.services.embeddings import EmbeddingService


@pytest.fixture
def embedding_service(test_settings):
    """Create EmbeddingService with test settings."""
    return EmbeddingService(test_settings)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI embeddings response with 2 embeddings."""
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1] * 1536),
        Mock(embedding=[0.2] * 1536)
    ]
    return mock_response


@pytest.fixture
def mock_openai_response_three():
    """Mock OpenAI embeddings response with 3 embeddings."""
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1] * 1536),
        Mock(embedding=[0.2] * 1536),
        Mock(embedding=[0.3] * 1536)
    ]
    return mock_response


@pytest.fixture
def mock_chat_response():
    """Mock OpenAI chat completion response."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content="This chunk discusses the main topic of the document."))
    ]
    return mock_response


@pytest.mark.asyncio
async def test_create_embeddings_batch_success(embedding_service, mock_openai_response) -> None:
    """Test successful creation of a batch of embeddings."""
    texts = ["hello", "world"]
    with patch.object(embedding_service.embedding_client.embeddings, 'create', return_value=mock_openai_response):
        embeddings = await embedding_service.create_embeddings_batch(texts)
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1] * 1536


@pytest.mark.asyncio
async def test_create_embeddings_batch_empty_list(embedding_service) -> None:
    """Test embedding creation with empty list."""
    embeddings = await embedding_service.create_embeddings_batch([])
    assert embeddings == []


@pytest.mark.asyncio
async def test_create_embeddings_batch_rate_limit_retry(embedding_service) -> None:
    """Test retry logic on rate limit errors."""
    texts = ["hello", "world"]
    
    # Create a proper mock response for rate limit error
    mock_response = Mock()
    mock_response.request = Mock()
    
    # Mock the create method to simulate a rate limit error on the first call
    mock_create = Mock(
        side_effect=[
            openai.RateLimitError("Rate limit exceeded", response=mock_response, body=None),
            # Create custom mock response with 2 embeddings
            SimpleNamespace(data=[
                SimpleNamespace(embedding=[0.1] * 1536),
                SimpleNamespace(embedding=[0.4] * 1536)
            ])
        ]
    )
    with patch.object(embedding_service.embedding_client.embeddings, 
                      'create',
                      new=mock_create):
        embeddings = await embedding_service.create_embeddings_batch(texts)
        assert len(embeddings) == 2
        assert embeddings[1] == [0.4] * 1536
        assert embedding_service.embedding_client.embeddings.create.call_count == 2


@pytest.mark.asyncio
async def test_create_embeddings_batch_fallback_to_individual(embedding_service) -> None:
    """Test fallback to individual embeddings when batch fails."""
    texts = ["hello", "world", "foo"]
    
    # Mock the create method to fail on all 3 batch retries, then succeed on individual calls
    mock_individual_responses = [
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536)]),
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.4] * 1536)]),
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.7] * 1536)])
    ]
    
    def side_effect_func(*args, **kwargs):
        if hasattr(side_effect_func, 'call_count'):
            side_effect_func.call_count += 1
        else:
            side_effect_func.call_count = 1
            
        # First 3 calls are batch retries (all fail)
        if side_effect_func.call_count <= 3:
            raise Exception("Batch failed")
        else:
            # Individual calls succeed (calls 4, 5, 6)
            return mock_individual_responses[side_effect_func.call_count - 4]
    
    with patch.object(embedding_service.embedding_client.embeddings,
                      'create',
                      side_effect=side_effect_func):

        embeddings = await embedding_service.create_embeddings_batch(texts)

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1] * 1536
        assert embeddings[1] == [0.4] * 1536
        assert embeddings[2] == [0.7] * 1536


@pytest.mark.asyncio
async def test_create_embeddings_batch_partial_failure(embedding_service) -> None:
    """Test partial failure where some individual embeddings also fail."""
    texts = ["success", "fail"]
    
    def side_effect_func(*args, **kwargs):
        if hasattr(side_effect_func, 'call_count'):
            side_effect_func.call_count += 1
        else:
            side_effect_func.call_count = 1
            
        # First 3 calls are batch retries (all fail)
        if side_effect_func.call_count <= 3:
            raise Exception("Batch failed")
        elif side_effect_func.call_count == 4:
            # First individual call succeeds
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536)])
        else:
            # Second individual call fails
            raise Exception("Individual embedding failed")
    
    with patch.object(embedding_service.embedding_client.embeddings, 'create', side_effect=side_effect_func):
        embeddings = await embedding_service.create_embeddings_batch(texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1] * 1536
        assert embeddings[1] == [0.0] * 1536


@pytest.mark.asyncio
async def test_create_embedding_success(embedding_service) -> None:
    """Test successful creation of a single embedding."""
    mock_response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 1536)])
    with patch.object(embedding_service.embedding_client.embeddings, 'create', return_value=mock_response):
        embedding = await embedding_service.create_embedding("test text")
        assert embedding == [0.1] * 1536


@pytest.mark.asyncio
async def test_create_embedding_failure(embedding_service) -> None:
    """Test failure to create a single embedding."""
    with patch.object(embedding_service.embedding_client.embeddings, 'create', side_effect=Exception("API error")):
        embedding = await embedding_service.create_embedding("test text")
        assert embedding == [0.0] * 1536


@pytest.mark.asyncio
async def test_generate_contextual_embedding_success(embedding_service, mock_chat_response) -> None:
    """Test successful contextual embedding generation."""
    with patch.object(embedding_service.client.chat.completions, 'create', return_value=mock_chat_response):
        full_doc = "This is a document about Python programming..."
        chunk = "Functions in Python are defined using the def keyword."
        
        contextual_text, was_contextualized = await embedding_service.generate_contextual_embedding(
            full_doc, chunk
        )
        
        assert was_contextualized is True
        assert "This chunk discusses the main topic of the document." in contextual_text
        assert chunk in contextual_text
        assert "---" in contextual_text


@pytest.mark.asyncio
async def test_generate_contextual_embedding_failure(embedding_service) -> None:
    """Test contextual embedding generation failure."""
    with patch.object(
        embedding_service.client.chat.completions,
        'create',
        side_effect=Exception("API error")
    ):
        full_doc = "This is a document about Python programming..."
        chunk = "Functions in Python are defined using the def keyword."
        
        contextual_text, was_contextualized = await embedding_service.generate_contextual_embedding(
            full_doc, chunk
        )
        
        assert was_contextualized is False
        assert contextual_text == chunk  # Should return original chunk on failure


@pytest.mark.asyncio
async def test_process_chunks_with_context_success(embedding_service, mock_chat_response) -> None:
    """Test processing multiple chunks with context."""
    with patch.object(embedding_service.client.chat.completions, 'create', return_value=mock_chat_response):
        chunks = [
            ("url1", "chunk1", "full_doc1"),
            ("url2", "chunk2", "full_doc2"),
            ("url3", "chunk3", "full_doc3")
        ]
        
        results = await embedding_service.process_chunks_with_context(chunks)
        
        assert len(results) == 3
        for contextual_text, was_contextualized in results:
            assert was_contextualized is True
            assert "This chunk discusses the main topic of the document." in contextual_text


@pytest.mark.asyncio
async def test_process_chunks_with_context_mixed_results(embedding_service) -> None:
    """Test processing chunks with mixed success/failure."""
    # Create mock responses that fail based on the chunk content (deterministic)
    mock_success_response = Mock()
    mock_success_response.choices = [
        Mock(message=Mock(content="This chunk discusses the main topic of the document."))
    ]
    
    def create_side_effect(*args, **kwargs):
        # Check the content of the message to determine if it should fail
        messages = kwargs.get('messages', [])
        if len(messages) > 1:
            user_message = messages[1].get('content', '')
            # Fail specifically for chunk2
            if 'chunk2' in user_message:
                raise Exception("API error")
        return mock_success_response
    
    with patch.object(
        embedding_service.client.chat.completions,
        'create',
        side_effect=create_side_effect
    ):
        chunks = [
            ("url1", "chunk1", "full_doc1"),
            ("url2", "chunk2", "full_doc2"),
            ("url3", "chunk3", "full_doc3")
        ]
        
        results = await embedding_service.process_chunks_with_context(chunks)
        
        assert len(results) == 3
        # First chunk succeeded
        assert results[0][1] is True
        assert "This chunk discusses the main topic of the document." in results[0][0]
        # Second chunk failed - should return original chunk
        assert results[1][1] is False
        assert results[1][0] == "chunk2"
        # Third chunk succeeded
        assert results[2][1] is True


def test_process_chunk_with_context_sync(embedding_service, mock_chat_response) -> None:
    """Test synchronous wrapper for chunk processing."""
    with patch.object(embedding_service.client.chat.completions, 'create', return_value=mock_chat_response):
        args = ("url", "chunk content", "full document")
        contextual_text, was_contextualized = embedding_service.process_chunk_with_context(args)
        
        assert was_contextualized is True
        assert "chunk content" in contextual_text


@pytest.mark.asyncio
async def test_long_document_truncation(embedding_service, mock_chat_response) -> None:
    """Test that long documents are truncated properly."""
    with patch.object(embedding_service.client.chat.completions, 'create', return_value=mock_chat_response) as mock_create:
        # Create a very long document
        full_doc = "x" * 30000
        chunk = "test chunk"
        
        await embedding_service.generate_contextual_embedding(full_doc, chunk)
        
        # Check that the prompt was created with truncated document
        call_args = mock_create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        # Document should be truncated to 25000 characters
        assert len(user_message) < 26000  # Some overhead for prompt template