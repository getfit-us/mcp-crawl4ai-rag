"""Tests for crawl cancellation tools."""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tools.cancel_crawl import cancel_crawl, cancel_all_crawls, get_active_crawls


@pytest.fixture
def mock_context():
    """Mock MCP context."""
    context = Mock()
    context.request_context = Mock()
    context.request_context.lifespan_context = Mock()
    context.request_context.lifespan_context.crawler = AsyncMock()
    context.request_context.lifespan_context.settings = Mock()
    return context


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
async def test_cancel_crawl_success(mock_service_class, mock_context):
    """Test successful crawl cancellation."""
    mock_service = Mock()
    mock_service.cancel_crawl_operation = AsyncMock(return_value=True)
    mock_service_class.return_value = mock_service
    
    result = await cancel_crawl(mock_context, "crawl-123")
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    assert result_data["crawl_id"] == "crawl-123"
    assert "cancelled" in result_data["message"]
    
    # Verify the service was called correctly
    mock_service.cancel_crawl_operation.assert_called_once_with("crawl-123")


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
async def test_cancel_crawl_not_found(mock_service_class, mock_context):
    """Test cancelling a non-existent crawl."""
    mock_service = Mock()
    mock_service.cancel_crawl_operation = AsyncMock(return_value=False)
    mock_service_class.return_value = mock_service
    
    result = await cancel_crawl(mock_context, "crawl-999")
    result_data = json.loads(result)
    
    assert result_data["success"] is False
    assert result_data["crawl_id"] == "crawl-999"
    assert "not found" in result_data["error"]


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
async def test_cancel_crawl_exception(mock_service_class, mock_context):
    """Test handling of exceptions during cancellation."""
    mock_service = Mock()
    mock_service.cancel_crawl_operation = AsyncMock(side_effect=Exception("Network error"))
    mock_service_class.return_value = mock_service
    
    result = await cancel_crawl(mock_context, "crawl-123")
    result_data = json.loads(result)
    
    assert result_data["success"] is False
    assert "Network error" in result_data["error"]
    assert result_data["crawl_id"] == "crawl-123"


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
async def test_cancel_all_crawls_success(mock_service_class, mock_context):
    """Test successful cancellation of all crawls."""
    mock_service = Mock()
    mock_service.cancel_all_crawl_operations = AsyncMock(return_value=2)
    mock_service_class.return_value = mock_service
    
    result = await cancel_all_crawls(mock_context)
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    assert result_data["cancelled_count"] == 2
    assert "Cancelled 2" in result_data["message"]
    
    # Verify the service was called correctly
    mock_service.cancel_all_crawl_operations.assert_called_once()


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
async def test_cancel_all_crawls_exception(mock_service_class, mock_context):
    """Test handling of exceptions during cancel all."""
    mock_service = Mock()
    mock_service.cancel_all_crawl_operations = AsyncMock(side_effect=Exception("Database error"))
    mock_service_class.return_value = mock_service
    
    result = await cancel_all_crawls(mock_context)
    result_data = json.loads(result)
    
    assert result_data["success"] is False
    assert "Database error" in result_data["error"]


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
@patch('asyncio.get_event_loop')
async def test_get_active_crawls_success(mock_event_loop, mock_service_class, mock_context):
    """Test getting active crawls information."""
    # Mock asyncio.get_event_loop().time() for consistent runtime calculation
    mock_loop = Mock()
    mock_loop.time.return_value = 1234567910.0
    mock_event_loop.return_value = mock_loop
    
    mock_service = Mock()
    mock_service.get_active_crawl_operations = AsyncMock(return_value={
        "crawl-123": {
            "operation_type": "recursive",
            "urls": ["https://example.com"],
            "start_time": 1234567890.0,
            "status": "running"
        },
        "crawl-456": {
            "operation_type": "sitemap",
            "urls": ["https://example.com/page1", "https://example.com/page2"],
            "start_time": 1234567900.0,
            "status": "running"
        }
    })
    mock_service_class.return_value = mock_service
    
    result = await get_active_crawls(mock_context)
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    assert result_data["total_active"] == 2
    assert "crawl-123" in result_data["active_crawls"]
    assert "crawl-456" in result_data["active_crawls"]
    
    # Check computed fields
    crawl_123 = result_data["active_crawls"]["crawl-123"]
    assert crawl_123["operation_type"] == "recursive"
    assert crawl_123["url_count"] == 1
    assert crawl_123["runtime_seconds"] == 20.0  # 1234567910.0 - 1234567890.0
    
    crawl_456 = result_data["active_crawls"]["crawl-456"]
    assert crawl_456["operation_type"] == "sitemap"
    assert crawl_456["url_count"] == 2
    assert crawl_456["runtime_seconds"] == 10.0  # 1234567910.0 - 1234567900.0
    
    # Verify the service was called correctly
    mock_service.get_active_crawl_operations.assert_called_once()


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
async def test_get_active_crawls_empty(mock_service_class, mock_context):
    """Test getting active crawls when none are running."""
    mock_service = Mock()
    mock_service.get_active_crawl_operations = AsyncMock(return_value={})
    mock_service_class.return_value = mock_service
    
    result = await get_active_crawls(mock_context)
    result_data = json.loads(result)
    
    assert result_data["success"] is True
    assert result_data["total_active"] == 0
    assert result_data["active_crawls"] == {}
    assert "Found 0" in result_data["message"]


@pytest.mark.asyncio
@patch('tools.cancel_crawl.CrawlingService')
async def test_get_active_crawls_exception(mock_service_class, mock_context):
    """Test handling of exceptions during get active crawls."""
    mock_service = Mock()
    mock_service.get_active_crawl_operations = AsyncMock(side_effect=Exception("Service error"))
    mock_service_class.return_value = mock_service
    
    result = await get_active_crawls(mock_context)
    result_data = json.loads(result)
    
    assert result_data["success"] is False
    assert "Service error" in result_data["error"] 