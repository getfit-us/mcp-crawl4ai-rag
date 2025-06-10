"""Tool for crawling a single page and storing in the database."""

import logging
from typing import Any, Dict
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from crawl4ai_mcp.models import CrawlContext, CrawlResult, CrawlType
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.services.database import DatabaseService

logger = logging.getLogger(__name__)

app = FastMCP("Crawl4AI RAG MCP Server")


@app.tool(name="crawl_single_page")
async def crawl_single_page(
    url: str,
    context: CrawlContext
) -> Dict[str, Any]:
    """
    Crawl a single web page and store its content in PostgreSQL.
    
    This tool crawls a single web page, extracts its text content, 
    The content is stored in PostgreSQL for later retrieval and querying.
    
    Args:
        url: The URL of the web page to crawl
        context: The crawling context with database connection
        
    Returns:
        Summary of the crawling operation and storage in PostgreSQL
    """
    try:
        # Get the database connection pool from context
        postgres_pool = context.supabase_client  # Field name kept for compatibility
        settings = context.settings
        
        # Initialize services
        database_service = DatabaseService(postgres_pool, settings)
        crawling_service = CrawlingService(context.crawler, database_service, settings)
        
        # Parse domain for source identification
        domain = urlparse(url).netloc
        
        logger.info(f"Starting crawl of single page: {url}")
        
        # Crawl the single page
        result = await crawling_service.crawl_single_page(url)
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Unknown error during crawling"),
                "pages_crawled": 0,
                "chunks_stored": 0
            }
        
        logger.info(f"Successfully crawled and stored page: {url}")
        
        return {
            "success": True,
            "url": url,
            "source": domain,
            "pages_crawled": 1,
            "chunks_stored": result.get("chunks_stored", 0),
            "code_examples_stored": result.get("code_examples_stored", 0),
            "crawl_type": CrawlType.SINGLE_PAGE.value
        }
        
    except Exception as e:
        logger.error(f"Error crawling single page {url}: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "pages_crawled": 0,
            "chunks_stored": 0
        }