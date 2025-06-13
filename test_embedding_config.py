#!/usr/bin/env python3
"""Test embedding service configuration with custom URLs."""

import sys
sys.path.insert(0, 'src')

from crawl4ai_mcp.services.embeddings import EmbeddingService
from types import SimpleNamespace
from unittest.mock import Mock, patch
import openai
import pytest
from src.config import Settings

@pytest.fixture
def settings_instance():
    """Create a mock settings object for testing."""
    return Settings(
        openai_api_key="test_api_key",
        postgres_user="test_user",
        postgres_password="test_password"
    )

def test_embedding_service_config():
    """
    Test that the embedding service correctly uses the custom embedding model URL
    when provided in the settings.
    """
    # Mock the settings to simulate having a custom embedding URL
    with patch('src.services.embeddings.get_settings') as mock_get_settings:
        mock_settings = SimpleNamespace(
            openai_api_key="test_api_key",
            openai_base_url="https://api.openai.com/v1",
            custom_embedding_url="http://localhost:8080/embed",
            embedding_model="custom-embedding-model",
            openai_organization=None
        )
        mock_get_settings.return_value = mock_settings

        # Mock the OpenAI client to inspect its initialization
        with patch('src.services.embeddings.openai.OpenAI') as mock_openai:
            # Import the service to trigger initialization
            from src.services.embeddings import EmbeddingService
            
            # The service should be initialized with two clients
            service = EmbeddingService()

            # Check that the embedding client was called with the custom URL
            embedding_client_kwargs = mock_openai.call_args_list[1][1]
            assert embedding_client_kwargs['base_url'] == "http://localhost:8080/embed"
            print("Embedding client correctly used the custom URL.")

def test_embedding_service_initialization(settings_instance):
    """Test that the EmbeddingService initializes correctly."""
    service = EmbeddingService(settings=settings_instance)
    assert service.settings == settings_instance
    assert isinstance(service.client, openai.OpenAI)

async def test_create_embedding(settings_instance):
    """Test that a single embedding can be created."""
    # Ensure the embedding model is set to a standard, reliable model
    settings_instance.embedding_model = "text-embedding-3-small"
    
    service = EmbeddingService(settings=settings_instance)
    embedding = await service.create_embedding("test")
    
    assert isinstance(embedding, list)
    assert len(embedding) == settings_instance.embedding_dimensions
    assert all(isinstance(x, float) for x in embedding)

if __name__ == '__main__':
    test_embedding_service_config() 