#!/usr/bin/env python3
"""Entry point for MCP Crawl4AI RAG server."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main() -> None:
    """Run the MCP server."""
    # Import will be available after creating mcp_server.py
    from src.mcp_server import run_server
    
    asyncio.run(run_server())


if __name__ == "__main__":
    main()