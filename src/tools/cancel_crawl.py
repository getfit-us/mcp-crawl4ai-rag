"""Tools for cancelling and managing crawl operations."""

import json
import logging
from typing import Optional

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.models import CrawlContext

logger = logging.getLogger(__name__)


@mcp.tool()
async def cancel_crawl(ctx: Context, crawl_id: str) -> str:
    """
    Cancel a specific crawl operation.
    
    This tool allows you to cancel an ongoing crawl operation using its unique ID.
    The crawl operation will be stopped as soon as possible, though some operations
    may continue briefly while the cancellation signal is processed.
    
    Args:
        ctx: The MCP server provided context
        crawl_id: The unique ID of the crawl operation to cancel
    
    Returns:
        JSON response indicating success or failure of the cancellation
    """
    try:
        # Get context and services
        context: CrawlContext = ctx.request_context.lifespan_context
        
        # Initialize crawling service
        crawling_service = CrawlingService(
            crawler=context.crawler, 
            settings=context.settings
        )
        
        # Attempt to cancel the crawl operation
        cancelled = await crawling_service.cancel_crawl_operation(crawl_id)
        
        if cancelled:
            return json.dumps({
                "success": True,
                "message": f"Crawl operation {crawl_id} has been cancelled",
                "crawl_id": crawl_id
            })
        else:
            return json.dumps({
                "success": False,
                "error": f"Crawl operation {crawl_id} not found or already completed",
                "crawl_id": crawl_id
            })
        
    except Exception as e:
        logger.error(f"Error cancelling crawl {crawl_id}: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "crawl_id": crawl_id
        })


@mcp.tool()
async def cancel_all_crawls(ctx: Context) -> str:
    """
    Cancel all active crawl operations.
    
    This tool cancels all currently running crawl operations. Use with caution
    as this will stop all ongoing crawls across the system.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON response indicating the number of operations cancelled
    """
    try:
        # Get context and services
        context: CrawlContext = ctx.request_context.lifespan_context
        
        # Initialize crawling service
        crawling_service = CrawlingService(
            crawler=context.crawler, 
            settings=context.settings
        )
        
        # Cancel all active crawl operations
        cancelled_count = await crawling_service.cancel_all_crawl_operations()
        
        return json.dumps({
            "success": True,
            "message": f"Cancelled {cancelled_count} crawl operations",
            "cancelled_count": cancelled_count
        })
        
    except Exception as e:
        logger.error(f"Error cancelling all crawls: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@mcp.tool()
async def get_active_crawls(ctx: Context) -> str:
    """
    Get information about all active crawl operations.
    
    This tool returns details about currently running crawl operations,
    including their IDs, types, start times, and status.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON response with information about active crawl operations
    """
    try:
        # Get context and services
        context: CrawlContext = ctx.request_context.lifespan_context
        
        # Initialize crawling service
        crawling_service = CrawlingService(
            crawler=context.crawler, 
            settings=context.settings
        )
        
        # Get active crawl operations
        active_crawls = await crawling_service.get_active_crawl_operations()
        
        # Format the response with additional computed fields
        formatted_crawls = {}
        current_time = None
        
        if active_crawls:
            import asyncio
            current_time = asyncio.get_event_loop().time()
        
        for crawl_id, info in active_crawls.items():
            runtime = current_time - info["start_time"] if current_time else 0
            formatted_crawls[crawl_id] = {
                **info,
                "runtime_seconds": round(runtime, 2),
                "url_count": len(info.get("urls", []))
            }
        
        return json.dumps({
            "success": True,
            "active_crawls": formatted_crawls,
            "total_active": len(active_crawls),
            "message": f"Found {len(active_crawls)} active crawl operations"
        })
        
    except Exception as e:
        logger.error(f"Error getting active crawls: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }) 