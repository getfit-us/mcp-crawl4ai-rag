"""Tests for search service with PostgreSQL backend."""

import pytest
import asyncpg
import json
import os
import numpy as np
from typing import AsyncGenerator
from unittest.mock import Mock

# Fix import paths to match actual project structure
from crawl4ai_mcp.services.search import SearchService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.models import SearchRequest, SearchResult, SearchType

# Set up the test environment variable for OpenAI API key
os.environ["OPENAI_API_KEY"] = "test-key"

# Get database URL from environment/config
def get_test_database_url() -> str:
    """Get the test database URL from PostgreSQL environment variables."""
    # Use individual PostgreSQL environment variables, defaulting to test database
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = "test_crawl4ai"  # Always use test database for tests
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    sslmode = os.getenv("POSTGRES_SSLMODE", "prefer")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode={sslmode}"


@pytest.fixture
async def postgres_pool() -> AsyncGenerator[asyncpg.Pool, None]:
    """Create a connection pool to the test PostgreSQL database."""
    database_url = get_test_database_url()
    print(f"Connecting to PostgreSQL: {database_url}")  # Debug output
    
    try:
        # Define connection initialization function to register vector types and JSONB codec
        async def init_connection(conn):
            from pgvector.asyncpg import register_vector
            await register_vector(conn)
            # Register JSONB codec to automatically parse JSON
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )
        
        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=5,
            init=init_connection
        )
        
        # Ensure tables and functions exist
        async with pool.acquire() as conn:
            await setup_test_schema(conn)
        
        yield pool
        
        # Cleanup
        await pool.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")
        return


async def setup_test_schema(conn: asyncpg.Connection) -> None:
    """Set up test schema in the database."""
    # Enable pgvector extension
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create sources table (matching setup_postgres.sql exactly)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS sources ( 
            source_id text PRIMARY KEY,
            summary text,
            total_word_count integer DEFAULT 0,
            created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
            updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
        )
    """)
    
    # Create crawled_pages table (matching setup_postgres.sql exactly)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS crawled_pages (
            id bigserial PRIMARY KEY,
            url varchar NOT NULL,
            chunk_number integer NOT NULL,
            content text NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            source_id text NOT NULL,
            embedding vector(1536),
            created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
            
            UNIQUE(url, chunk_number),
            FOREIGN KEY (source_id) REFERENCES sources(source_id)
        )
    """)
    
    # Create code_examples table (matching setup_postgres.sql exactly)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS code_examples (
            id bigserial PRIMARY KEY,
            url varchar NOT NULL,
            chunk_number integer NOT NULL,
            content text NOT NULL,
            summary text NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            source_id text NOT NULL,
            embedding vector(1536),
            created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
            
            UNIQUE(url, chunk_number),
            FOREIGN KEY (source_id) REFERENCES sources(source_id)
        )
    """)
    
    # Create search functions
    await conn.execute("""
        CREATE OR REPLACE FUNCTION match_crawled_pages (
          query_embedding vector(1536),
          match_count int DEFAULT 10,
          filter jsonb DEFAULT '{}'::jsonb,
          source_filter text DEFAULT NULL
        ) RETURNS TABLE (
          id bigint,
          url varchar,
          chunk_number integer,
          content text,
          metadata jsonb,
          source_id text,
          similarity float
        )
        LANGUAGE plpgsql
        AS $$
        #variable_conflict use_column
        BEGIN
          RETURN QUERY
          SELECT
            crawled_pages.id,
            crawled_pages.url,
            crawled_pages.chunk_number,
            crawled_pages.content,
            crawled_pages.metadata,
            crawled_pages.source_id,
            1 - (crawled_pages.embedding <=> query_embedding) as similarity
          FROM crawled_pages
          WHERE (filter = '{}'::jsonb OR crawled_pages.metadata @> filter)
            AND (source_filter IS NULL OR crawled_pages.source_id = source_filter)
          ORDER BY crawled_pages.embedding <=> query_embedding
          LIMIT match_count;
        END;
        $$;
    """)
    
    await conn.execute("""
        CREATE OR REPLACE FUNCTION match_code_examples (
          query_embedding vector(1536),
          match_count int DEFAULT 10,
          filter jsonb DEFAULT '{}'::jsonb,
          source_filter text DEFAULT NULL
        ) RETURNS TABLE (
          id bigint,
          url varchar,
          chunk_number integer,
          content text,
          summary text,
          metadata jsonb,
          source_id text,
          similarity float
        )
        LANGUAGE plpgsql
        AS $$
        #variable_conflict use_column
        BEGIN
          RETURN QUERY
          SELECT
            code_examples.id,
            code_examples.url,
            code_examples.chunk_number,
            code_examples.content,
            code_examples.summary,
            code_examples.metadata,
            code_examples.source_id,
            1 - (code_examples.embedding <=> query_embedding) as similarity
          FROM code_examples
          WHERE (filter = '{}'::jsonb OR code_examples.metadata @> filter)
            AND (source_filter IS NULL OR code_examples.source_id = source_filter)
          ORDER BY code_examples.embedding <=> query_embedding
          LIMIT match_count;
        END;
        $$;
    """)


@pytest.fixture
async def clean_database(postgres_pool: asyncpg.Pool) -> None:
    """Clean the database before each test."""
    async with postgres_pool.acquire() as conn:
        await conn.execute("DELETE FROM code_examples")
        await conn.execute("DELETE FROM crawled_pages")
        await conn.execute("DELETE FROM sources")


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service that returns 1536-dimension embeddings for PostgreSQL."""
    service = Mock(spec=EmbeddingService)
    # Make it an async mock - PostgreSQL uses 1536 dimensions
    # Return an embedding that will match well with our test data
    async def mock_create_embedding(text):
        return np.array([0.9] + [0.1] * 1535, dtype=np.float32)  # High similarity with test_embedding_1
    service.create_embedding = mock_create_embedding
    return service


@pytest.fixture
def test_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.use_agentic_rag = True
    settings.use_reranking = False
    settings.default_rerank_threshold = 0.3
    return settings


@pytest.fixture
def search_service(postgres_pool: asyncpg.Pool, test_settings, mock_embedding_service):
    """Create SearchService with PostgreSQL connection pool."""
    # Pass test_settings explicitly to avoid config validation
    return SearchService(postgres_pool, test_settings, mock_embedding_service)


async def insert_test_data(conn: asyncpg.Connection) -> None:
    """Insert test data for search tests."""
    # Insert source
    await conn.execute("""
        INSERT INTO sources (source_id, summary, total_word_count)
        VALUES ('example.com', 'Test source', 100)
        ON CONFLICT (source_id) DO NOTHING
    """)
    
    # Insert test documents with embeddings that match the mock embedding service
    # The mock always returns [0.9] + [0.1] * 1535, so use the same for test data
    mock_embedding = np.array([0.9] + [0.1] * 1535, dtype=np.float32)
    test_embedding_2 = np.array([0.8] + [0.1] * 1535, dtype=np.float32)  # Medium similarity  
    test_embedding_3 = np.array([0.7] + [0.1] * 1535, dtype=np.float32)  # Lower similarity
    
    await conn.execute("""
        INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
        VALUES 
            ($1, 1, $2, $3, $4, $5),
            ($6, 2, $7, $8, $9, $10),
            ($11, 3, $12, $13, $14, $15)
        ON CONFLICT (url, chunk_number) DO NOTHING
    """, 
        "https://example.com/1", "First result content", {"title": "First"}, "example.com", mock_embedding,
        "https://example.com/2", "Second result content", {"title": "Second"}, "example.com", test_embedding_2,
        "https://example.com/3", "Third result content", {"title": "Third"}, "example.com", test_embedding_3
    )
    
    # Insert test code example with the same mock embedding for best match  
    await conn.execute("""
        INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
        VALUES ($1, 1, $2, $3, $4, $5, $6)
        ON CONFLICT (url, chunk_number) DO NOTHING
    """, 
        "https://example.com/code1", 
        'def example():\n    return "Hello"',
        "Example function",
        {"language": "python"},
        "example.com",
        mock_embedding
    )


@pytest.mark.asyncio
async def test_search_documents_success(search_service: SearchService, clean_database, postgres_pool: asyncpg.Pool) -> None:
    """Test successful document search with PostgreSQL."""
    # Insert test data
    async with postgres_pool.acquire() as conn:
        await insert_test_data(conn)
    
    results = await search_service.search_documents(
        query="test query",
        match_count=10
    )
    
    assert len(results) == 3
    assert isinstance(results[0], SearchResult)
    
    # Results should be ordered by similarity (highest first)
    assert results[0].content == "First result content"
    assert results[1].content == "Second result content"
    assert results[2].content == "Third result content"
    
    # Check similarity scores (cosine similarity with our test embeddings)
    assert results[0].similarity_score > results[1].similarity_score
    assert results[1].similarity_score > results[2].similarity_score


@pytest.mark.asyncio
async def test_search_documents_with_filters(search_service: SearchService, clean_database, postgres_pool: asyncpg.Pool) -> None:
    """Test document search with metadata filters."""
    # Insert test data with specific metadata
    async with postgres_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO sources (source_id, summary) 
            VALUES ('docs.example.com', 'Test docs source')
            ON CONFLICT (source_id) DO NOTHING
        """)
        
        test_embedding = np.array([0.9] + [0.1] * 1535, dtype=np.float32)
        await conn.execute("""
            INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
            VALUES ($1, 1, $2, $3, $4, $5)
            ON CONFLICT (url, chunk_number) DO NOTHING
        """, 
            "https://docs.example.com/tutorial", 
            "Tutorial content",
            {"category": "tutorial", "level": "beginner"},
            "docs.example.com",
            test_embedding
        )
    
    # Search with metadata filter
    results = await search_service.search_documents(
        query="test query",
        match_count=5,
        filter_metadata={"category": "tutorial"},
        source_id="docs.example.com"
    )
    
    assert len(results) == 1
    assert results[0].content == "Tutorial content"
    assert results[0].source == "docs.example.com"
    assert results[0].metadata["category"] == "tutorial"


@pytest.mark.asyncio
async def test_search_code_examples_success(search_service: SearchService, clean_database, postgres_pool: asyncpg.Pool) -> None:
    """Test successful code example search."""
    # Insert test data
    async with postgres_pool.acquire() as conn:
        await insert_test_data(conn)
    
    results = await search_service.search_code_examples(
        query="example function",
        language="python",
        match_count=5
    )
    
    assert len(results) == 1
    assert results[0]['content'] == 'def example():\n    return "Hello"'
    assert results[0]['summary'] == "Example function"
    assert results[0]['metadata']['language'] == 'python'
    assert results[0]['source_id'] == 'example.com'


@pytest.mark.asyncio
async def test_perform_search_documents_only(search_service: SearchService, clean_database, postgres_pool: asyncpg.Pool) -> None:
    """Test perform_search with documents only."""
    # Insert test data
    async with postgres_pool.acquire() as conn:
        await insert_test_data(conn)
    
    request = SearchRequest(
        query="test query",
        num_results=5,
        semantic_threshold=0.85  # Only high similarity results
    )
    
    response = await search_service.perform_search(request)
    
    assert response.success is True
    # Only results with similarity >= 0.85 should be included
    assert len(response.results) >= 1  # At least the first high-similarity result
    assert response.search_type == SearchType.SEMANTIC
    
    # Verify all results meet the threshold
    for result in response.results:
        assert result.similarity_score >= 0.85


@pytest.mark.asyncio
async def test_perform_search_with_code_examples(search_service: SearchService, clean_database, postgres_pool: asyncpg.Pool, test_settings) -> None:
    """Test perform_search including code examples."""
    # Enable code examples
    test_settings.use_agentic_rag = True
    
    # Insert test data
    async with postgres_pool.acquire() as conn:
        await insert_test_data(conn)
    
    request = SearchRequest(query="test query", num_results=5)
    
    response = await search_service.perform_search(
        request,
        include_code_examples=True
    )
    
    assert response.success is True
    assert len(response.results) >= 4  # 3 documents + 1 code example (minimum)
    
    # Verify code example is included
    # The code examples come from the match_code_examples function and have language metadata
    code_results = [r for r in response.results if r.metadata.get('language') == 'python']
    assert len(code_results) >= 1
    
    code_result = code_results[0]
    assert code_result.metadata['language'] == 'python'


@pytest.mark.asyncio
async def test_vector_similarity_calculation(clean_database, postgres_pool: asyncpg.Pool) -> None:
    """Test that vector similarity calculations work correctly."""
    async with postgres_pool.acquire() as conn:
        # Insert test documents with known embeddings
        await conn.execute("""
            INSERT INTO sources (source_id, summary) 
            VALUES ('test.com', 'Test source')
            ON CONFLICT (source_id) DO NOTHING
        """)
        
        # Create embeddings with known relationships (convert to numpy arrays)
        # Use normalized vectors for better cosine similarity results
        identical_embedding = np.array([1.0] + [0.0] * 1535, dtype=np.float32)
        similar_embedding = np.array([0.99] + [0.01] * 1535, dtype=np.float32)  # Very similar
        different_embedding = np.array([0.0] + [0.0] * 1534 + [1.0], dtype=np.float32)  # Different
        
        await conn.execute("""
            INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
            VALUES 
                ($1, 1, $2, $3, $4, $5),
                ($6, 2, $7, $8, $9, $10),
                ($11, 3, $12, $13, $14, $15)
            ON CONFLICT (url, chunk_number) DO NOTHING
        """, 
            "https://test.com/identical", "Identical content", {}, "test.com", identical_embedding,
            "https://test.com/similar", "Similar content", {}, "test.com", similar_embedding,
            "https://test.com/different", "Different content", {}, "test.com", different_embedding
        )
        
        # Search with identical embedding
        results = await conn.fetch("""
            SELECT url, content, similarity
            FROM match_crawled_pages($1, 10, '{}'::jsonb, NULL)
            ORDER BY similarity DESC
        """, identical_embedding)
        
        assert len(results) == 3
        # Identical embedding should have highest similarity (close to 1.0)
        assert results[0]['url'] == "https://test.com/identical"
        assert results[0]['similarity'] > 0.99
        
        # Similar embedding should be second
        assert results[1]['url'] == "https://test.com/similar"
        assert results[1]['similarity'] > 0.92  # Adjusted for very similar vectors
        
        # Different embedding should have lowest similarity
        assert results[2]['url'] == "https://test.com/different"
        assert results[2]['similarity'] < 0.2  # Adjusted for orthogonal vectors 


def test_database_url_configuration():
    """Test that database URL is properly configured from environment variables."""
    # Test with default values
    url = get_test_database_url()
    assert "postgresql://" in url
    assert "localhost" in url or os.getenv("POSTGRES_HOST") in url
    assert "test_crawl4ai" in url  # Always uses test database
    
    # Test with custom environment variables (if set)
    if os.getenv("POSTGRES_HOST"):
        assert os.getenv("POSTGRES_HOST") in url
    if os.getenv("POSTGRES_USER"):
        assert os.getenv("POSTGRES_USER") in url


def test_embedding_dimensions():
    """Test that PostgreSQL tests use 1536-dimension embeddings."""
    mock_service = Mock(spec=EmbeddingService)
    async def mock_create_embedding(text):
        return np.array([0.1] * 1536, dtype=np.float32)
    mock_service.create_embedding = mock_create_embedding
    
    # Verify the mock returns 1536 dimensions
    import asyncio
    embedding = asyncio.run(mock_service.create_embedding("test"))
    assert len(embedding) == 1536 