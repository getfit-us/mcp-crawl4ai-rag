"""MCP server setup and initialization."""

import json
import logging
import signal
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg
from crawl4ai import AsyncWebCrawler, BrowserConfig
from mcp.server.fastmcp import FastMCP
from pgvector.asyncpg import register_vector
from sentence_transformers import CrossEncoder

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.utilities.browser_cleanup import emergency_browser_cleanup
from crawl4ai_mcp.utilities.browser_monitor import start_browser_monitoring, stop_browser_monitoring

logger = logging.getLogger(__name__)

# Global state for signal handling
_shutdown_event: Optional[asyncio.Event] = None
_current_context: Optional[CrawlContext] = None


async def _handle_shutdown_signal(signum: int) -> None:
    """Handle shutdown signals gracefully."""
    signal_name = signal.Signals(signum).name
    logger.warning(f"Received signal {signal_name} ({signum}), initiating graceful shutdown...")
    
    # Set shutdown event
    global _shutdown_event
    if _shutdown_event:
        _shutdown_event.set()
    
    # Perform emergency cleanup if we have a current context
    global _current_context
    if _current_context:
        try:
            logger.info("Performing emergency browser cleanup due to signal...")
            await emergency_browser_cleanup()
        except Exception as e:
            logger.error(f"Emergency cleanup during signal handling failed: {e}")


def _setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown."""
    try:
        # Only set up signal handlers on Unix systems
        if hasattr(signal, 'SIGTERM'):
            def signal_handler(signum: int, frame: Any) -> None:
                """Signal handler that schedules async cleanup."""
                logger.warning(f"Received signal {signum}, scheduling cleanup...")
                try:
                    # Try to get the running event loop
                    loop = asyncio.get_running_loop()
                    # Schedule the async cleanup
                    loop.create_task(_handle_shutdown_signal(signum))
                except RuntimeError:
                    # No running loop, run emergency cleanup in sync mode
                    logger.error("No async loop available, performing basic cleanup")
                    import sys
                    sys.exit(1)
            
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, signal_handler)
            logger.debug("Signal handlers installed for graceful shutdown")
    except Exception as e:
        logger.warning(f"Could not set up signal handlers: {e}")


def _cleanup_signal_handlers() -> None:
    """Clean up signal handlers."""
    try:
        if hasattr(signal, 'SIGTERM'):
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, signal.SIG_DFL)
            logger.debug("Signal handlers cleaned up")
    except Exception as e:
        logger.warning(f"Could not clean up signal handlers: {e}")


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
    global _shutdown_event, _current_context
    
    # Initialize shutdown event
    _shutdown_event = asyncio.Event()
    
    # Set up signal handlers for graceful shutdown
    _setup_signal_handlers()
    
    settings = get_settings()
    
    # Use default browser configuration, let Crawl4AI handle all browser management
    browser_config = BrowserConfig(
        headless=getattr(settings, 'browser_headless', True),
        verbose=False,
        # Let Crawl4AI manage all browser settings and timeouts
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
    
    # Store context globally for signal handling
    _current_context = context
    
    # Start browser health monitoring if auto-cleanup is enabled
    if settings.browser_auto_cleanup:
        try:
            await start_browser_monitoring(
                check_interval=300,  # 5 minutes
                max_processes_threshold=20,
                auto_cleanup_orphaned=True
            )
            logger.info("Browser health monitoring started")
        except Exception as e:
            logger.warning(f"Failed to start browser monitoring: {e}")
    
    try:
        logger.info("MCP server lifespan started, browser and database initialized")
        yield context
    finally:
        # Robust cleanup with timeout protection
        cleanup_success = True
        
        # Step 1: Clean up crawler with timeout
        try:
            logger.info("Starting browser cleanup...")
            cleanup_task = asyncio.create_task(crawler.__aexit__(None, None, None))
            await asyncio.wait_for(cleanup_task, timeout=settings.browser_cleanup_timeout)
            logger.info("Browser cleanup completed successfully")
        except asyncio.TimeoutError:
            logger.error(f"Browser cleanup timed out after {settings.browser_cleanup_timeout}s")
            cleanup_success = False
            # Cancel the cleanup task
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            logger.error(f"Error during browser cleanup: {e}")
            cleanup_success = False
        
        # Step 2: Emergency browser cleanup if needed
        if not cleanup_success and settings.browser_auto_cleanup:
            try:
                logger.warning("Performing emergency browser cleanup due to failed graceful cleanup")
                await asyncio.wait_for(emergency_browser_cleanup(), timeout=10)
            except Exception as e:
                logger.error(f"Emergency browser cleanup failed: {e}")
        
        # Step 3: Clean up PostgreSQL pool (separate from browser cleanup)
        try:
            logger.debug("Closing PostgreSQL connection pool...")
            await postgres_pool.close()
            logger.debug("PostgreSQL connection pool closed successfully")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL pool: {e}")
        
        # Step 4: Stop browser monitoring
        if settings.browser_auto_cleanup:
            try:
                logger.debug("Stopping browser health monitoring...")
                await stop_browser_monitoring()
                logger.debug("Browser health monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping browser monitoring: {e}")
        
        # Step 5: Clean up signal handlers and global state
        _cleanup_signal_handlers()
        _current_context = None
        _shutdown_event = None
        
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



