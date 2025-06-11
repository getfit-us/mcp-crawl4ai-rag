#!/usr/bin/env python3
"""Test embedding service configuration with custom URLs."""

import sys
sys.path.insert(0, 'src')

from crawl4ai_mcp.services.embeddings import EmbeddingService
from types import SimpleNamespace
from unittest.mock import Mock, patch

def test_embedding_service_config():
    """Test embedding service configuration with custom settings."""
    print("Testing embedding service configuration...")
    
    # Test 1: Default OpenAI configuration
    settings1 = SimpleNamespace(
        openai_api_key='test-key',
        openai_base_url=None,
        openai_organization=None,
        embedding_model='text-embedding-3-small',
        embedding_dimensions=1536
    )
    
    with patch('crawl4ai_mcp.services.embeddings.openai.OpenAI') as MockOpenAI:
        service1 = EmbeddingService(settings1)
        MockOpenAI.assert_called_once_with(api_key='test-key')
        print('✓ Default OpenAI configuration works')
    
    # Test 2: Custom base URL
    settings2 = SimpleNamespace(
        openai_api_key='test-key',
        openai_base_url='https://my-custom-endpoint.com/v1',
        openai_organization=None,
        embedding_model='my-custom-model',
        embedding_dimensions=768
    )
    
    with patch('crawl4ai_mcp.services.embeddings.openai.OpenAI') as MockOpenAI:
        service2 = EmbeddingService(settings2)
        MockOpenAI.assert_called_once_with(
            api_key='test-key',
            base_url='https://my-custom-endpoint.com/v1'
        )
        print('✓ Custom base URL configuration works')
    
    # Test 3: Custom base URL and organization
    settings3 = SimpleNamespace(
        openai_api_key='test-key',
        openai_base_url='https://api.azure.openai.com/',
        openai_organization='org-12345',
        embedding_model='text-embedding-ada-002',
        embedding_dimensions=1536
    )
    
    with patch('crawl4ai_mcp.services.embeddings.openai.OpenAI') as MockOpenAI:
        service3 = EmbeddingService(settings3)
        MockOpenAI.assert_called_once_with(
            api_key='test-key',
            base_url='https://api.azure.openai.com/',
            organization='org-12345'
        )
        print('✓ Custom base URL and organization configuration works')
    
    # Test 4: Only organization (no custom URL)
    settings4 = SimpleNamespace(
        openai_api_key='test-key',
        openai_base_url=None,
        openai_organization='org-67890',
        embedding_model='text-embedding-3-large',
        embedding_dimensions=3072
    )
    
    with patch('crawl4ai_mcp.services.embeddings.openai.OpenAI') as MockOpenAI:
        service4 = EmbeddingService(settings4)
        MockOpenAI.assert_called_once_with(
            api_key='test-key',
            organization='org-67890'
        )
        print('✓ Organization-only configuration works')
    
    print('\nAll embedding service configuration tests passed! ✓')
    print('The custom embedding model functionality is working correctly.')

if __name__ == '__main__':
    test_embedding_service_config() 