"""Tests for database service."""

import pytest
from unittest.mock import Mock

from src.services.database import DatabaseService
from src.models import SourceInfo


@pytest.fixture
def database_service(mock_supabase_client, test_settings):
    """Create DatabaseService with mocked dependencies."""
    return DatabaseService(mock_supabase_client, test_settings)


@pytest.mark.asyncio
async def test_add_documents_success(database_service, mock_supabase_client):
    """Test successful document addition."""
    # Mock the delete operation
    mock_supabase_client.table.return_value.delete.return_value.in_.return_value.execute.return_value = Mock()
    
    # Mock the insert operation
    mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock()
    
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


@pytest.mark.asyncio
async def test_add_documents_empty_list(database_service):
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
async def test_add_documents_delete_error(database_service, mock_supabase_client):
    """Test handling of delete errors."""
    # Mock delete to raise an exception
    mock_supabase_client.table.return_value.delete.return_value.in_.return_value.execute.side_effect = Exception("Delete failed")
    
    result = await database_service.add_documents(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        contents=["Test content"],
        embeddings=[[0.1] * 1536],
        metadatas=[{"title": "Test"}],
        url_to_full_document={}
    )
    
    assert result["success"] is False
    assert "Failed to delete existing records" in result["error"]
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_add_code_examples_success(database_service, mock_supabase_client):
    """Test successful code example addition."""
    # Mock the delete operation
    mock_supabase_client.table.return_value.delete.return_value.eq.return_value.execute.return_value = Mock()
    
    # Mock the insert operation
    mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock()
    
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


@pytest.mark.asyncio
async def test_add_code_examples_empty_list(database_service):
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
async def test_update_source_info_new_source(database_service, mock_supabase_client):
    """Test creating a new source."""
    # Mock update to return no data (source doesn't exist)
    mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(data=[])
    
    # Mock insert
    mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock()
    
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Test source",
        word_count=100
    )
    
    assert result["success"] is True
    assert result["source_id"] == "example.com"
    
    # Verify insert was called
    mock_supabase_client.table.return_value.insert.assert_called_once()


@pytest.mark.asyncio
async def test_update_source_info_existing_source(database_service, mock_supabase_client):
    """Test updating an existing source."""
    # Mock update to return data (source exists)
    mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(data=[{"source_id": "example.com"}])
    
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Updated source",
        word_count=200
    )
    
    assert result["success"] is True
    assert result["source_id"] == "example.com"
    
    # Verify update was called
    mock_supabase_client.table.return_value.update.assert_called_once()


@pytest.mark.asyncio
async def test_update_source_info_error(database_service, mock_supabase_client):
    """Test handling of source update errors."""
    # Mock update to raise an exception
    mock_supabase_client.table.return_value.update.return_value.eq.return_value.execute.side_effect = Exception("Update failed")
    
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Test source",
        word_count=100
    )
    
    assert result["success"] is False
    assert "Update failed" in result["error"]


@pytest.mark.asyncio
async def test_get_available_sources(database_service, mock_supabase_client):
    """Test getting available sources."""
    # Mock sources table response
    mock_sources_data = [
        {
            "source_id": "example.com",
            "summary": "Example website",
            "total_word_count": 1000,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z"
        }
    ]
    
    # Create separate mocks for each table operation
    sources_table_mock = Mock()
    sources_table_mock.select.return_value.execute.return_value = Mock(data=mock_sources_data)
    
    crawled_pages_table_mock = Mock()
    # For count query
    count_result = Mock()
    count_result.count = 10
    crawled_pages_table_mock.select.return_value.eq.return_value.execute.return_value = count_result
    
    code_examples_table_mock = Mock()
    # For count query
    code_count_result = Mock()
    code_count_result.count = 5
    code_examples_table_mock.select.return_value.eq.return_value.execute.return_value = code_count_result
    
    # Setup table method to return different mocks
    call_count = 0
    def table_side_effect(table_name):
        nonlocal call_count
        call_count += 1
        
        if table_name == 'sources':
            return sources_table_mock
        elif table_name == 'crawled_pages':
            if call_count in [2, 5]:  # Document count calls
                return crawled_pages_table_mock
            else:  # Chunk data call
                chunks_mock = Mock()
                chunks_mock.select.return_value.eq.return_value.execute.return_value = Mock(
                    data=[{"chunk_number": 1}, {"chunk_number": 2}, {"chunk_number": 3}]
                )
                return chunks_mock
        elif table_name == 'code_examples':
            return code_examples_table_mock
            
        return Mock()
    
    mock_supabase_client.table.side_effect = table_side_effect
    
    sources = await database_service.get_available_sources()
    
    assert len(sources) == 1
    assert isinstance(sources[0], SourceInfo)
    assert sources[0].source == "example.com"
    assert sources[0].summary == "Example website"
    assert sources[0].word_count == 1000
    assert sources[0].total_documents == 10
    assert sources[0].total_code_examples == 5
    assert sources[0].total_chunks == 3


@pytest.mark.asyncio
async def test_get_available_sources_error(database_service, mock_supabase_client):
    """Test handling of errors when getting sources."""
    # Mock to raise an exception
    mock_supabase_client.table.return_value.select.return_value.execute.side_effect = Exception("Query failed")
    
    sources = await database_service.get_available_sources()
    
    assert sources == []


@pytest.mark.asyncio
async def test_generate_contextual_content(database_service):
    """Test contextual content generation."""
    content = database_service._generate_contextual_content(
        chunk_content="This is chunk content",
        full_document="This is the full document with more content",
        chunk_number=2
    )
    
    assert "Chunk 2" in content
    assert "This is chunk content" in content