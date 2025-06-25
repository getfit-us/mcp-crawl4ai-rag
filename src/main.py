#!/usr/bin/env python3
"""Entry point for MCP Crawl4AI RAG server."""

import logging

# Configure logging
logger = logging.getLogger('mcp-crawl4ai-rag')
logger.setLevel(logging.INFO)

# Create handler with formatter
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)


def main() -> None:
    """Run the MCP server."""
    from crawl4ai_mcp.mcp_server import get_mcp_server
    from crawl4ai_mcp.config import get_settings
    
    # Import tools here to avoid circular imports
    from crawl4ai_mcp.tools.crawl_single_page import crawl_single_page  # noqa: F401
    from crawl4ai_mcp.tools.smart_crawl_url import smart_crawl_url  # noqa: F401
    from crawl4ai_mcp.tools.get_available_sources import get_available_sources  # noqa: F401
    from crawl4ai_mcp.tools.perform_rag_query import perform_rag_query  # noqa: F401
    from crawl4ai_mcp.tools.search_code_examples import search_code_examples  # noqa: F401
    from crawl4ai_mcp.tools.cancel_crawl import cancel_crawl, cancel_all_crawls, get_active_crawls  # noqa: F401
    from crawl4ai_mcp.tools.browser_health import get_browser_status, cleanup_browser_processes, get_browser_configuration  # noqa: F401
    
    settings = get_settings()
    server = get_mcp_server()
    
    logger.info(f"Starting MCP server on {settings.host}:{settings.port}")
    logger.info(f"Transport: {settings.transport}")
    logger.warning("Note: FastMCP 2.8.1 uses its own default host/port settings")
    
    # Use the synchronous run() method which handles the async context internally
    # Note: FastMCP 2.8.1 doesn't accept host/port as parameters, it uses its own defaults
    if settings.transport == "stdio":
        server.run(transport="stdio")
    elif settings.transport == "sse":
        server.run(transport="sse")
    else:
        # streamable-http transport (default)
        server.run(transport="streamable-http")


if __name__ == "__main__":
    main()