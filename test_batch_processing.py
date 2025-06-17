#!/usr/bin/env python3
"""
Test script to demonstrate batch processing capabilities.

This script shows how the new environment variables control batch processing
for embeddings and summaries.
"""

import asyncio
import logging
import os
import time
from typing import List

# Configure environment variables for batch processing tests
def setup_test_environment():
    """Set up all environment variables for batch processing tests."""
    # Batch embeddings configuration
    os.environ["ENABLE_BATCH_EMBEDDINGS"] = "true"
    os.environ["EMBEDDING_BATCH_SIZE"] = "20"
    os.environ["EMBEDDING_MODEL"] = "Q3-Embed"
    os.environ["EMBEDDING_DIMENSIONS"] = "1536"  # Standard embedding dimensions
    
    # Fix the embedding endpoint to use the correct OpenAI-compatible path
    # The OpenAI client will append "/embeddings" to this URL, so we need "/v1"
    os.environ["CUSTOM_EMBEDDING_URL"] = "http://192.168.1.244:8080/v1"
    os.environ["EMBEDDING_API_KEY"] = "123"
    
    # Batch summaries configuration
    os.environ["ENABLE_BATCH_SUMMARIES"] = "true"
    os.environ["SUMMARY_BATCH_SIZE"] = "20"
    
    # Batch contextual embeddings configuration
    os.environ["ENABLE_BATCH_CONTEXTUAL_EMBEDDINGS"] = "true"
    os.environ["CONTEXTUAL_EMBEDDING_BATCH_SIZE"] = "20"

# Set up environment before imports
setup_test_environment()

# Add src to path for imports
import sys
sys.path.append('src')

from crawl4ai_mcp.config import get_settings, reset_settings
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.services.crawling import CrawlingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reset and get settings after environment is configured
reset_settings()  # Clear any cached settings
SETTINGS = get_settings()


def generate_test_data(qty: int = 5):
    """Generate test data for batch processing tests.

    Args:
        qty: Number of test items to generate

    Returns:
        Dict with all test data structures
    """
    # Generate base texts
    base_texts = [f"This is test text {i+1} for batch processing." for i in range(qty)]

    # Generate code examples data
    code_examples = [
        (
            f"def test_func_{i}():\\n    print('Test {i} code')",
            f"Example {i}:",
            f"Description of example {i}",
        )
        for i in range(1, qty + 1)
    ]

    # Generate contextual embedding data
    context_documents = [
        f"""
    This is document {i+1} for contextual embeddings.
    It contains information about Python programming and data processing.
    The document is structured to demonstrate contextual embedding generation.
    """
        for i in range(qty)
    ]

    # Create chunks with documents
    contextual_data = [
        (f"Contextual chunk {i+1} about Python", context_documents[i])
        for i in range(qty)
    ]

    return {
        "texts": base_texts,
        "code_examples": code_examples,
        "contextual_data": contextual_data,
    }


async def test_batch_embeddings():
    """Test batch embedding processing."""
    print("\n=== Testing Batch Embeddings ===")
    
    embedding_service = EmbeddingService(SETTINGS)
    
    # Test data
    test_data = generate_test_data(40)
    texts = test_data["texts"]
    
    print(f"Processing {len(texts)} texts with batch processing ENABLED")
    print(f"Batch size: {SETTINGS.embedding_batch_size}")
    
    start_time = time.time()
    
    try:
        # This should use batch processing
        embeddings = await embedding_service.create_embeddings_batch(texts)
        
        batch_time = time.time() - start_time
        print(f"Batch processing took: {batch_time:.2f} seconds")
        print(f"Generated {len(embeddings)} embeddings")
        print(f"First embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
        
        return batch_time
        
    except Exception as e:
        logger.error(f"Error in batch embedding test: {e}")
        return None

async def test_individual_embeddings():
    """Test individual embedding processing."""
    print("\n=== Testing Individual Embeddings ===")
    
    # For this test, we'll temporarily disable batch processing
    original_batch_setting = os.environ.get("ENABLE_BATCH_EMBEDDINGS")
    os.environ["ENABLE_BATCH_EMBEDDINGS"] = "false"
    
    # Get new settings with batch disabled
    individual_settings = get_settings()
    embedding_service = EmbeddingService(individual_settings)
    
    # Test data
    texts = [
        "This is the first text to embed.",
        "Here is another piece of text for embedding.",
        "The third text demonstrates individual processing.",
        "Fourth text for testing individual functionality.",
        "Fifth text to complete our test set."
    ]
    
    print(f"Processing {len(texts)} texts with batch processing DISABLED")
    
    start_time = time.time()
    
    try:
        # Process individually
        embeddings = []
        for text in texts:
            embedding = await embedding_service.create_embedding(text)
            embeddings.append(embedding)
        
        individual_time = time.time() - start_time
        print(f"Individual processing took: {individual_time:.2f} seconds")
        print(f"Generated {len(embeddings)} embeddings")
        print(f"First embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
        
        return individual_time
        
    except Exception as e:
        logger.error(f"Error in individual embedding test: {e}")
        return None
    finally:
        # Restore original setting
        if original_batch_setting is not None:
            os.environ["ENABLE_BATCH_EMBEDDINGS"] = original_batch_setting
        else:
            os.environ.pop("ENABLE_BATCH_EMBEDDINGS", None)

async def test_batch_summaries():
    """Test batch summary processing."""
    print("\n=== Testing Batch Summaries ===")
    
    embedding_service = EmbeddingService(SETTINGS)
    crawling_service = CrawlingService(None, SETTINGS, embedding_service)
    
    # Test data for code example summaries
    test_data = generate_test_data(40)
    code_examples_data = test_data["code_examples"]
    
    print(f"Processing {len(code_examples_data)} code summaries with batch processing ENABLED")
    print(f"Batch size: {SETTINGS.summary_batch_size}")
    
    start_time = time.time()
    
    try:
        summaries = await crawling_service.generate_code_example_summaries_batch(code_examples_data)
        
        batch_time = time.time() - start_time
        print(f"Batch summary processing took: {batch_time:.2f} seconds")
        print(f"Generated {len(summaries)} summaries")
        for i, summary in enumerate(summaries):
            print(f"Summary {i+1}: {summary[:100]}...")
        
        return batch_time
        
    except Exception as e:
        logger.error(f"Error in batch summary test: {e}")
        return None

async def test_contextual_batch_embeddings():
    """Test batch contextual embedding processing."""
    print("\n=== Testing Batch Contextual Embeddings ===")
    
    embedding_service = EmbeddingService(SETTINGS)
    
    # Test data
    test_data = generate_test_data(40)
    chunks_with_docs = test_data["contextual_data"]
    
    print(f"Processing {len(chunks_with_docs)} contextual embeddings with batch processing ENABLED")
    print(f"Batch size: {SETTINGS.contextual_embedding_batch_size}")
    
    start_time = time.time()
    
    try:
        results = await embedding_service.generate_contextual_embeddings_batch(chunks_with_docs)
        
        batch_time = time.time() - start_time
        print(f"Batch contextual processing took: {batch_time:.2f} seconds")
        print(f"Generated {len(results)} contextual embeddings")
        for i, (contextual_text, was_contextualized) in enumerate(results):
            print(f"Result {i+1}: {'Contextualized' if was_contextualized else 'Original'}")
            print(f"  Text preview: {contextual_text[:100]}...")
        
        return batch_time
        
    except Exception as e:
        logger.error(f"Error in batch contextual embedding test: {e}")
        return None

async def main():
    """Run all batch processing tests."""
    print("🚀 Testing Batch Processing Implementation")
    print("=" * 50)
    
    # Test environment variables
    print("\n📊 Environment Variables:")
    batch_vars = [
        "ENABLE_BATCH_EMBEDDINGS",
        "EMBEDDING_BATCH_SIZE", 
        "ENABLE_BATCH_SUMMARIES",
        "SUMMARY_BATCH_SIZE",
        "ENABLE_BATCH_CONTEXTUAL_EMBEDDINGS",
        "CONTEXTUAL_EMBEDDING_BATCH_SIZE"
    ]
    
    for var in batch_vars:
        value = os.environ.get(var, "not set")
        print(f"  {var}: {value}")
    
    # Run tests
    tests = [
        ("Batch Embeddings", test_batch_embeddings),
        ("Individual Embeddings", test_individual_embeddings),
        ("Batch Summaries", test_batch_summaries),
        ("Batch Contextual Embeddings", test_contextual_batch_embeddings)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            results[test_name] = None
    
    # Summary
    print("\n📈 Performance Summary:")
    print("=" * 30)
    
    for test_name, duration in results.items():
        if duration is not None:
            print(f"{test_name}: {duration:.2f}s")
        else:
            print(f"{test_name}: FAILED")
    
    # Compare batch vs individual embeddings
    if results.get("Batch Embeddings") and results.get("Individual Embeddings"):
        batch_time = results["Batch Embeddings"]
        individual_time = results["Individual Embeddings"]
        speedup = individual_time / batch_time if batch_time > 0 else 0
        print(f"\n🏆 Embedding Speedup: {speedup:.1f}x faster with batch processing")
    
    print("\n✅ Batch processing tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 
