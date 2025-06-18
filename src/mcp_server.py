"""MCP server setup and initialization."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from crawl4ai import AsyncWebCrawler, BrowserConfig
from mcp.server.fastmcp import FastMCP
from pgvector.asyncpg import register_vector
from sentence_transformers import CrossEncoder

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import CrawlContext

logger = logging.getLogger(__name__)


async def _init_postgres_connection(conn):
    await register_vector(conn)
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def get_postgres_pool() -> asyncpg.Pool:
    """
    Get PostgreSQL connection pool with configuration from settings.
    
    Returns:
        asyncpg.Pool: Configured PostgreSQL connection pool
    """
    settings = get_settings()
    return await asyncpg.create_pool(
        settings.postgres_dsn,
        min_size=1,
        max_size=10,
        command_timeout=60,
        init=_init_postgres_connection,
    )


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[CrawlContext]:
    """
    Manage the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        CrawlContext: The context containing the Crawl4AI crawler and PostgreSQL pool
    """
    settings = get_settings()
    
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize PostgreSQL connection pool
    postgres_pool = await get_postgres_pool()
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if settings.use_reranking:
        try:
            reranking_model = CrossEncoder(settings.cross_encoder_model)
        except Exception as e:
            logger.warning(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    # Create context
    context = CrawlContext(
        crawler=crawler,
        supabase_client=postgres_pool,  # Reusing the field name for compatibility
        reranking_model=reranking_model,
        settings=settings
    )
    
    try:
        yield context
    finally:
        # Cleanup
        await crawler.__aexit__(None, None, None)
        await postgres_pool.close()


# Initialize FastMCP server
def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    settings = get_settings()
    
    return FastMCP(
        "mcp-crawl4ai-rag",
        description="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan,
        host=settings.host,
        port=settings.port
    )


# Global server instance - lazy initialization
_mcp_server = None


def get_mcp_server() -> FastMCP:
    """Get or create the MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = create_mcp_server()
    return _mcp_server


# For backward compatibility, create a property-like access
class MCPProxy:
    def __getattr__(self, name):
        return getattr(get_mcp_server(), name)

mcp = MCPProxy()


async def run_server() -> None:
    """Run the MCP server."""
    settings = get_settings()
    
    # Import tools here to avoid circular imports
    from crawl4ai_mcp.tools.crawl_single_page import crawl_single_page  # noqa: F401
    from crawl4ai_mcp.tools.smart_crawl_url import smart_crawl_url  # noqa: F401
    from crawl4ai_mcp.tools.get_available_sources import get_available_sources  # noqa: F401
    from crawl4ai_mcp.tools.perform_rag_query import perform_rag_query  # noqa: F401
    from crawl4ai_mcp.tools.search_code_examples import search_code_examples  # noqa: F401
    from crawl4ai_mcp.tools.cancel_crawl import cancel_crawl, cancel_all_crawls, get_active_crawls  # noqa: F401
    
    logger.info(f"Starting MCP server on {settings.host}:{settings.port}")
    logger.info(f"Transport: {settings.transport}")
    
    # Get the MCP server instance
    server = get_mcp_server()
    
    # Run based on transport
    if settings.transport == "stdio":
        await server.run_stdio_async()
    elif settings.transport == "sse":
        await server.run(transport="sse", host=settings.host, port=settings.port)
    else:
        # streamable-http transport (default)
        await server.run(transport="streamable-http", host=settings.host, port=settings.port)