"""MCP server setup and initialization."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from crawl4ai import AsyncWebCrawler, BrowserConfig
from mcp.server.fastmcp import FastMCP
from pgvector.asyncpg import register_vector
from sentence_transformers import CrossEncoder

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import CrawlContext

logger = logging.getLogger(__name__)


async def _init_postgres_connection(conn: Any) -> None:
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
    
    # Create browser configuration - let Crawl4AI handle all browser management
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
            reranking_model = CrossEncoder(settings.reranker_model)
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
        logger.info("MCP server lifespan started, browser and database initialized")
        yield context
    finally:
        # Simple cleanup - let Crawl4AI handle its own browser management
        try:
            logger.info("Cleaning up crawler...")
            await crawler.__aexit__(None, None, None)
            logger.info("Crawler cleanup completed")
        except Exception as e:
            logger.error(f"Error during crawler cleanup: {e}")
        
        # Clean up PostgreSQL pool
        try:
            logger.debug("Closing PostgreSQL connection pool...")
            await postgres_pool.close()
            logger.debug("PostgreSQL connection pool closed")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL pool: {e}")
        
        logger.info("Lifespan cleanup completed")


# Initialize FastMCP server
def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server."""
    return FastMCP(
        "mcp-crawl4ai-rag",
        description="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan
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
    def __getattr__(self, name: str) -> Any:
        return getattr(get_mcp_server(), name)

mcp = MCPProxy()



