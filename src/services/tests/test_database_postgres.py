"""Tests for database service with local PostgreSQL PGVector deployment."""

import pytest
import asyncpg
import json
import os
from typing import AsyncGenerator
from unittest.mock import Mock

from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.models import SourceInfo

# Set fake OpenAI API key for testing
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


@pytest.fixture(scope="function")
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
        
        # Ensure tables exist
        async with pool.acquire() as conn:
            await setup_test_tables(conn)
        
        yield pool
        
        # Cleanup
        await pool.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")
        return


async def setup_test_tables(conn: asyncpg.Connection) -> None:
    """Set up test tables in the database (matching setup_postgres.sql exactly)."""
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
    
    # Create indexes (matching setup_postgres.sql exactly)
    await conn.execute("CREATE INDEX IF NOT EXISTS crawled_pages_embedding_idx ON crawled_pages USING ivfflat (embedding vector_cosine_ops)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_metadata ON crawled_pages USING gin (metadata)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_source_id ON crawled_pages (source_id)")
    
    await conn.execute("CREATE INDEX IF NOT EXISTS code_examples_embedding_idx ON code_examples USING ivfflat (embedding vector_cosine_ops)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_code_examples_metadata ON code_examples USING gin (metadata)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_code_examples_source_id ON code_examples (source_id)")


@pytest.fixture
async def clean_database(postgres_pool: asyncpg.Pool) -> None:
    """Clean the database before each test."""
    async with postgres_pool.acquire() as conn:
        await conn.execute("DELETE FROM code_examples")
        await conn.execute("DELETE FROM crawled_pages")
        await conn.execute("DELETE FROM sources")


@pytest.fixture
def test_settings():
    """Mock settings for testing."""
    return Mock()


@pytest.fixture
def database_service(postgres_pool: asyncpg.Pool, test_settings) -> DatabaseService:
    """Create DatabaseService with real PostgreSQL connection."""
    return DatabaseService(postgres_pool, test_settings)


@pytest.mark.asyncio
async def test_add_documents_success(database_service: DatabaseService, clean_database) -> None:
    """Test successful document addition."""
    # Create a test embedding (1536 dimensions to match our schema)
    test_embedding = [0.1] * 1536
    
    result = await database_service.add_documents(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        contents=["Test content"],
        embeddings=[test_embedding],
        metadatas=[{"title": "Test"}],
        url_to_full_document={"https://example.com/test": "Full document content"}
    )
    
    assert result["success"] is True
    assert result["count"] == 1
    assert result["total"] == 1
    
    # Verify the document was actually inserted
    async with database_service.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM crawled_pages WHERE url = $1", "https://example.com/test")
        assert len(rows) == 1
        assert rows[0]["content"] == "Test content"
        assert rows[0]["chunk_number"] == 1
        assert rows[0]["source_id"] == "example.com"


@pytest.mark.asyncio
async def test_add_documents_empty_list(database_service: DatabaseService, clean_database) -> None:
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
async def test_add_documents_multiple_batches(database_service: DatabaseService, clean_database) -> None:
    """Test adding documents in multiple batches."""
    # Create test data for multiple documents
    urls = [f"https://example.com/test{i}" for i in range(25)]  # More than default batch size
    chunk_numbers = list(range(1, 26))
    contents = [f"Test content {i}" for i in range(25)]
    embeddings = [[0.1] * 1536 for _ in range(25)]  # 1536 dimensions
    metadatas = [{"title": f"Test {i}"} for i in range(25)]
    url_to_full_document = {url: f"Full document {i}" for i, url in enumerate(urls)}
    
    result = await database_service.add_documents(
        urls=urls,
        chunk_numbers=chunk_numbers,
        contents=contents,
        embeddings=embeddings,
        metadatas=metadatas,
        url_to_full_document=url_to_full_document,
        batch_size=10  # Force multiple batches
    )
    
    assert result["success"] is True
    assert result["count"] == 25
    assert result["total"] == 25
    
    # Verify all documents were inserted
    async with database_service.pool.acquire() as conn:
        rows = await conn.fetch("SELECT COUNT(*) as count FROM crawled_pages")
        assert rows[0]["count"] == 25


@pytest.mark.asyncio
async def test_add_documents_replaces_existing(database_service: DatabaseService, clean_database) -> None:
    """Test that adding documents replaces existing ones for the same URL."""
    test_embedding = [0.1] * 1536  # 1536 dimensions
    url = "https://example.com/test"
    
    # Add initial document
    await database_service.add_documents(
        urls=[url],
        chunk_numbers=[1],
        contents=["Initial content"],
        embeddings=[test_embedding],
        metadatas=[{"title": "Initial"}],
        url_to_full_document={url: "Initial full document"}
    )
    
    # Add new document with same URL
    result = await database_service.add_documents(
        urls=[url],
        chunk_numbers=[1],
        contents=["Updated content"],
        embeddings=[test_embedding],
        metadatas=[{"title": "Updated"}],
        url_to_full_document={url: "Updated full document"}
    )
    
    assert result["success"] is True
    assert result["count"] == 1
    
    # Verify only the new document exists
    async with database_service.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM crawled_pages WHERE url = $1", url)
        assert len(rows) == 1
        assert rows[0]["content"] == "Updated content"


@pytest.mark.asyncio
async def test_add_code_examples_success(database_service: DatabaseService, clean_database) -> None:
    """Test successful code example addition."""
    test_embedding = [0.1] * 1536  # 1536 dimensions
    
    result = await database_service.add_code_examples(
        urls=["https://example.com/test"],
        chunk_numbers=[1],
        code_examples=["```python\ndef hello():\n    print('Hello')\n```"],
        summaries=["A hello function"],
        embeddings=[test_embedding],
        metadatas=[{"language": "python"}]
    )
    
    assert result["success"] is True
    assert result["count"] == 1
    assert result["total"] == 1
    
    # Verify the code example was inserted
    async with database_service.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM code_examples WHERE url = $1", "https://example.com/test")
        assert len(rows) == 1
        assert "def hello():" in rows[0]["content"]
        assert rows[0]["summary"] == "A hello function"
        assert rows[0]["metadata"]["language"] == "python"


@pytest.mark.asyncio
async def test_add_code_examples_empty_list(database_service: DatabaseService, clean_database) -> None:
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
async def test_update_source_info_new_source(database_service: DatabaseService, clean_database) -> None:
    """Test creating a new source."""
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Test source",
        word_count=100
    )
    
    assert result["success"] is True
    assert result["source_id"] == "example.com"
    
    # Verify the source was created
    async with database_service.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM sources WHERE source_id = $1", "example.com")
        assert len(rows) == 1
        assert rows[0]["summary"] == "Test source"
        assert rows[0]["total_word_count"] == 100


@pytest.mark.asyncio
async def test_update_source_info_existing_source(database_service: DatabaseService, clean_database) -> None:
    """Test updating an existing source."""
    # First create a source
    async with database_service.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO sources (source_id, summary, total_word_count) VALUES ($1, $2, $3)",
            "example.com", "Original summary", 50
        )
    
    # Update the source
    result = await database_service.update_source_info(
        source_id="example.com",
        summary="Updated source",
        word_count=200
    )
    
    assert result["success"] is True
    assert result["source_id"] == "example.com"
    
    # Verify the source was updated
    async with database_service.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM sources WHERE source_id = $1", "example.com")
        assert len(rows) == 1
        assert rows[0]["summary"] == "Updated source"
        assert rows[0]["total_word_count"] == 200


@pytest.mark.asyncio
async def test_get_available_sources(database_service: DatabaseService, clean_database) -> None:
    """Test getting available sources."""
    # Create test data
    async with database_service.pool.acquire() as conn:
        # Create source
        await conn.execute(
            "INSERT INTO sources (source_id, summary, total_word_count) VALUES ($1, $2, $3)",
            "example.com", "Example website", 1000
        )
        
        # Create documents
        test_embedding = [0.1] * 1536  # 1536 dimensions
        for i in range(3):
            await conn.execute(
                """INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding) 
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                f"https://example.com/page{i}", i + 1, f"Content {i}", 
                json.dumps({"title": f"Page {i}"}), "example.com", test_embedding
            )
        
        # Create code examples
        for i in range(2):
            await conn.execute(
                """INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                f"https://example.com/code{i}", i + 1, f"def func{i}(): pass", 
                f"Function {i}", json.dumps({"language": "python"}), "example.com", test_embedding
            )
    
    sources = await database_service.get_available_sources()
    
    assert len(sources) == 1
    assert isinstance(sources[0], SourceInfo)
    assert sources[0].source == "example.com"
    assert sources[0].summary == "Example website"
    assert sources[0].word_count == 1000
    assert sources[0].total_documents == 3  # 3 unique URLs
    assert sources[0].total_code_examples == 2
    assert sources[0].total_chunks == 3


@pytest.mark.asyncio
async def test_get_available_sources_empty(database_service: DatabaseService, clean_database) -> None:
    """Test getting sources when database is empty."""
    sources = await database_service.get_available_sources()
    assert sources == []


@pytest.mark.asyncio
async def test_generate_contextual_content(database_service: DatabaseService) -> None:
    """Test contextual content generation."""
    content = database_service._generate_contextual_content(
        chunk_content="This is chunk content",
        full_document="This is the full document with more content",
        chunk_number=2
    )
    
    assert "Chunk 2" in content
    assert "This is chunk content" in content


@pytest.mark.asyncio
async def test_database_with_real_vector_operations(database_service: DatabaseService, clean_database) -> None:
    """Test that vector operations work correctly with pgvector."""
    # Create embeddings with different values (1536 dimensions)
    embedding1 = [0.1] * 768 + [0.2] * 768  # 1536 dimensions
    embedding2 = [0.2] * 768 + [0.1] * 768  # Different embedding
    
    # Add documents with different embeddings
    await database_service.add_documents(
        urls=["https://example.com/doc1", "https://example.com/doc2"],
        chunk_numbers=[1, 1],
        contents=["Document 1 content", "Document 2 content"],
        embeddings=[embedding1, embedding2],
        metadatas=[{"title": "Doc 1"}, {"title": "Doc 2"}],
        url_to_full_document={}
    )
    
    # Verify embeddings were stored correctly
    async with database_service.pool.acquire() as conn:
        rows = await conn.fetch("SELECT url, embedding FROM crawled_pages ORDER BY url")
        assert len(rows) == 2
        
        # Check that embeddings are different
        emb1 = list(rows[0]["embedding"])
        emb2 = list(rows[1]["embedding"])
        assert emb1 != emb2
        assert len(emb1) == 1536  # 1536 dimensions
        assert len(emb2) == 1536  # 1536 dimensions 