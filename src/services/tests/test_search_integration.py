"""Tests for search service with a real PostgreSQL database."""

import pytest
import asyncpg
import os

# Make sure to load environment variables for the database connection
from dotenv import load_dotenv
load_dotenv()

from crawl4ai_mcp.services.search import SearchService
from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.mcp_server import get_postgres_pool


# This fixture is to ensure that the test is skipped if no real DB is configured
# It checks for a specific environment variable that the user should set to run this test.
requires_real_db = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") is None,
    reason="requires RUN_INTEGRATION_TESTS env var to be set"
)


@pytest.fixture(scope="function")
async def real_postgres_pool():
    """Create a real PostgreSQL connection pool for testing."""
    # Ensure settings are loaded
    get_settings()
    pool = await get_postgres_pool()
    yield pool
    await pool.close()


@pytest.fixture
def database_service(real_postgres_pool: asyncpg.Pool):
    """Create DatabaseService with a real connection pool."""
    return DatabaseService(pool=real_postgres_pool)


@pytest.fixture
def search_service(real_postgres_pool: asyncpg.Pool):
    """Create SearchService with a real connection pool and real embedding service."""
    settings = get_settings()
    # We are now using the real EmbeddingService to match against real data.
    embedding_service = EmbeddingService(settings=settings)
    return SearchService(pool=real_postgres_pool, settings=settings, embedding_service=embedding_service)


@requires_real_db
@pytest.mark.asyncio
async def test_integration_search_with_existing_data(
    search_service: SearchService,
):
    """
    Test searching for a document in the real database using existing data.
    This test is designed to debug issues with search results not appearing.

    To run this test:
    1. Make sure you have data in your `crawled_pages` table.
    2. Make sure your database schema is up to date.
    3. Create a .env file in the project root with your database credentials.
    4. Set the environment variable to enable this test:
       export RUN_INTEGRATION_TESTS=1
    5. Run pytest.
    """
    print("\n--- Running Integration Test: test_integration_search_with_existing_data ---")

    # 1. DEFINE YOUR SEARCH QUERY HERE
    # Please replace this with a query you expect to return results from your DB.
    search_query = "shadcn"

    # If the query is still the placeholder, we skip the test.
    if search_query == "YOUR_SEARCH_QUERY_HERE":
        pytest.skip("Please replace 'YOUR_SEARCH_QUERY_HERE' with a real query.")


    # 2. Search for the document
    search_results = await search_service.search_documents(
        query=search_query, match_count=5, source_id="ui.shadcn.com"
    )

    # 3. Assert that we get results
    print(f"Found {len(search_results)} results.")
    if len(search_results) > 0:
        print("Top result content:", search_results[0].content)
        print("Top result similarity:", search_results[0].similarity_score)

    assert len(search_results) > 0, "Search returned no results. This confirms the issue with your existing data." 
