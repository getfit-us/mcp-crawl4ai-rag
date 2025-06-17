"""Tests for database service."""

import pytest
from unittest.mock import Mock

from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.models import SourceInfo


@pytest.fixture
def database_service(mock_postgres_pool, test_settings):
    """Create DatabaseService with mocked dependencies."""
    return DatabaseService(mock_postgres_pool, test_settings)


@pytest.mark.asyncio
async def test_add_documents_success(database_service, mock_postgres_pool) -> None:
    """Test successful document addition."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Setup mock responses
    mock_conn.execute.return_value = None  # For INSERT/DELETE operations
    mock_conn.executemany.return_value = None  # For batch inserts
    
    result = await database_service.add_documents(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        contents=["Test content"],
        embeddings=[[0.1] * 1536],
        metadatas=[{"title": "Test"}],
        url_to_full_document={"https://example.com/test": "Full document content"}
    )
    
    assert result["success"] is True
    assert result["count"] == 1
    assert result["total"] == 1
    
    # Verify PostgreSQL operations were called
    mock_conn.execute.assert_called()  # Should be called for source creation and deletion
    mock_conn.executemany.assert_called()  # Should be called for batch insert


@pytest.mark.asyncio
async def test_add_documents_empty_list(database_service) -> None:
    """Test handling of empty document list."""
    result = await database_service.add_documents(
        urls=[],
        chunk_numbers=[],
        contents=[],
        embeddings=[],
        metadatas=[],
        url_to_full_document={}
    )
    
    assert result["success"] is True
    assert result["count"] == 0
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_add_documents_delete_error(database_service, mock_postgres_pool) -> None:
    """Test handling of delete errors."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock execute to raise an exception on source creation or deletion
    mock_conn.execute.side_effect = Exception("Delete failed")
    
    result = await database_service.add_documents(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        contents=["Test content"],
        embeddings=[[0.1] * 1536],
        metadatas=[{"title": "Test"}],
        url_to_full_document={}
    )
    
    assert result["success"] is False
    assert "Failed to create source records" in result["error"] or "Failed to delete existing records" in result["error"]
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_add_code_examples_success(database_service, mock_postgres_pool) -> None:
    """Test successful code example addition."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Setup mock responses
    mock_conn.execute.return_value = None  # For INSERT/DELETE operations
    mock_conn.executemany.return_value = None  # For batch inserts
    
    result = await database_service.add_code_examples(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        code_examples=["```python\ndef hello():\n    print('Hello')\n```"],
        summaries=["A hello function"],
        embeddings=[[0.1] * 1536],
        metadatas=[{"language": "python"}]
    )
    
    assert result["success"] is True
    assert result["count"] == 1
    assert result["total"] == 1
    
    # Verify PostgreSQL operations were called
    mock_conn.execute.assert_called()  # Should be called for source creation and deletion
    mock_conn.executemany.assert_called()  # Should be called for batch insert


@pytest.mark.asyncio
async def test_add_code_examples_empty_list(database_service) -> None:
    """Test handling of empty code examples list."""
    result = await database_service.add_code_examples(
        urls=[],
        chunk_numbers=[],
        code_examples=[],
        summaries=[],
        embeddings=[],
        metadatas=[]
    )
    
    assert result["success"] is True
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_update_source_info_new_source(database_service, mock_postgres_pool) -> None:
    """Test creating a new source."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock execute to return "UPDATE 0" (no rows updated) for the update, then succeed for insert
    mock_conn.execute.side_effect = ["UPDATE 0", None]
    
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Test source",
        word_count=100
    )
    
    assert result["success"] is True
    assert result["source_id"] == "example.com"
    
    # Verify operations were called (update attempt, then insert)
    assert mock_conn.execute.call_count == 2


@pytest.mark.asyncio
async def test_update_source_info_existing_source(database_service, mock_postgres_pool) -> None:
    """Test updating an existing source."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock execute to return "UPDATE 1" (one row updated)
    mock_conn.execute.return_value = "UPDATE 1"
    
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Updated source",
        word_count=200
    )
    
    assert result["success"] is True
    assert result["source_id"] == "example.com"
    
    # Verify update was called once
    mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_update_source_info_error(database_service, mock_postgres_pool) -> None:
    """Test handling of source update errors."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock execute to raise an exception
    mock_conn.execute.side_effect = Exception("Update failed")
    
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Test source",
        word_count=100
    )
    
    assert result["success"] is False
    assert "Update failed" in result["error"]


@pytest.mark.asyncio
async def test_get_available_sources(database_service, mock_postgres_pool) -> None:
    """Test getting available sources."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock the JOIN query response with proper column names
    mock_sources_data = [
        {
            "source_id": "example.com",
            "summary": "Example website",
            "total_word_count": 1000,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "doc_count": 10,
            "chunk_count": 25,
            "code_count": 5
        }
    ]
    
    # Mock fetch response for the JOIN query
    mock_conn.fetch.return_value = mock_sources_data
    
    sources = await database_service.get_available_sources()
    
    assert len(sources) == 1
    assert sources[0].source == "example.com"  # SourceInfo uses 'source' field
    assert sources[0].summary == "Example website"
    assert sources[0].word_count == 1000
    assert sources[0].total_documents == 10
    assert sources[0].total_chunks == 25
    assert sources[0].total_code_examples == 5
    
    # Verify PostgreSQL query was called once
    mock_conn.fetch.assert_called_once()


@pytest.mark.asyncio
async def test_get_available_sources_column_names(database_service, mock_postgres_pool) -> None:
    """Test that get_available_sources handles the column names correctly."""
    # Get the mock connection from the pool
    mock_conn = mock_postgres_pool._mock_conn
    
    # Mock sources data with specific column structure matching the JOIN query
    mock_sources_data = [
        {
            "source_id": "docs.example.com",
            "summary": "Documentation site",
            "total_word_count": 5000,
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-03T15:30:00Z",
            "doc_count": 15,
            "chunk_count": 45,
            "code_count": 3
        },
        {
            "source_id": "blog.example.com",
            "summary": "Blog posts",
            "total_word_count": 3000,
            "created_at": "2024-01-02T09:00:00Z",
            "updated_at": "2024-01-04T10:15:00Z",
            "doc_count": 8,
            "chunk_count": 20,
            "code_count": 2
        }
    ]
    
    # Mock fetch response
    mock_conn.fetch.return_value = mock_sources_data
    
    sources = await database_service.get_available_sources()
    
    assert len(sources) == 2
    
    # Check first source
    assert sources[0].source == "docs.example.com"
    assert sources[0].summary == "Documentation site"
    assert sources[0].word_count == 5000
    assert sources[0].total_documents == 15
    assert sources[0].total_chunks == 45
    assert sources[0].total_code_examples == 3
    
    # Check second source
    assert sources[1].source == "blog.example.com"
    assert sources[1].summary == "Blog posts"
    assert sources[1].word_count == 3000
    assert sources[1].total_documents == 8
    assert sources[1].total_chunks == 20
    assert sources[1].total_code_examples == 2


@pytest.mark.asyncio
async def test_generate_contextual_content(database_service) -> None:
    """Test contextual content generation."""
    content = database_service._generate_contextual_content(
        chunk_content="This is chunk content",
        full_document="This is the full document with more content",
        chunk_number=2
    )
    
    assert "Chunk 2" in content
    assert "This is chunk content" in content