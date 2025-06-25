"""Browser health monitoring tool for MCP server."""

import asyncio
import logging
from typing import Any, Dict
from datetime import datetime

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.utilities.browser_cleanup import (
    get_chrome_process_count,
    find_orphaned_chrome_processes,
    cleanup_orphaned_chrome_processes
)
from crawl4ai_mcp.utilities.browser_monitor import get_browser_monitoring_status

logger = logging.getLogger(__name__)


@mcp.tool()
def get_browser_status() -> Dict[str, Any]:
    """
    Get current browser process status and health information.
    
    Returns:
        Dict with browser status including process counts and monitoring info
    """
    try:
        # Get basic process information
        total_processes = get_chrome_process_count()
        orphaned_processes = find_orphaned_chrome_processes()
        
        # Get monitoring status (safely handle async in sync context)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, create a new task
                monitor_info = {"error": "Cannot get monitoring status from within async context"}
            else:
                monitor_info = loop.run_until_complete(get_browser_monitoring_status())
        except RuntimeError:
            # No event loop running, create one
            monitor_info = asyncio.run(get_browser_monitoring_status())
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "total_chrome_processes": total_processes,
            "orphaned_processes": len(orphaned_processes),
            "orphaned_details": [
                {
                    "pid": proc.pid,
                    "create_time": proc.create_time(),
                    "name": proc.name(),
                    "cmdline": proc.cmdline()[:3] if proc.cmdline() else []
                }
                for proc in orphaned_processes[:5]  # Limit to first 5 for readability
            ],
            "monitoring": monitor_info,
            "status": "healthy" if total_processes < 20 and len(orphaned_processes) == 0 else "warning"
        }
        
        warnings_list = []
        if total_processes > 20:
            warnings_list.append(f"High Chrome process count: {total_processes} (threshold: 20)")
        
        if len(orphaned_processes) > 0:
            warnings_list.append(f"Orphaned Chrome processes detected: {len(orphaned_processes)}")
        
        if warnings_list:
            result["warnings"] = warnings_list
            
        return result
        
    except Exception as e:
        logger.error(f"Error getting browser status: {e}")
        return {
            "error": f"Failed to get browser status: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@mcp.tool()
def cleanup_browser_processes() -> Dict[str, Any]:
    """
    Clean up orphaned browser processes manually.
    
    Returns:
        Dict with cleanup results
    """
    try:
        # Get initial count
        initial_count = len(find_orphaned_chrome_processes())
        
        if initial_count == 0:
            return {
                "success": True,
                "message": "No orphaned processes found to clean up",
                "processes_cleaned": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Perform cleanup (safely handle async in sync context)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, cannot run cleanup
                return {
                    "success": False,
                    "error": "Cannot perform cleanup from within async context - use async cleanup_browser_processes tool",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                cleaned_count = loop.run_until_complete(cleanup_orphaned_chrome_processes())
        except RuntimeError:
            # No event loop running, create one
            cleaned_count = asyncio.run(cleanup_orphaned_chrome_processes())
        
        return {
            "success": True,
            "message": f"Successfully cleaned up {cleaned_count} orphaned processes",
            "initial_orphaned_count": initial_count,
            "processes_cleaned": cleaned_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during browser cleanup: {e}")
        return {
            "success": False,
            "error": f"Failed to clean up browser processes: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@mcp.tool()
def get_browser_configuration() -> Dict[str, Any]:
    """
    Get current browser configuration and settings.
    
    Returns:
        Dict with browser configuration details
    """
    try:
        context = mcp.get_context()
        settings = context.settings
        
        return {
            "browser_headless": getattr(settings, 'browser_headless', True),
            "browser_timeout": getattr(settings, 'browser_timeout', 60),
            "browser_page_timeout": getattr(settings, 'browser_page_timeout', 30),
            "browser_navigation_timeout": getattr(settings, 'browser_navigation_timeout', 20),
            "browser_cleanup_timeout": getattr(settings, 'browser_cleanup_timeout', 10),
            "browser_max_memory_mb": getattr(settings, 'browser_max_memory_mb', 2048),
            "browser_process_isolation": getattr(settings, 'browser_process_isolation', True),
            "browser_auto_cleanup": getattr(settings, 'browser_auto_cleanup', True),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting browser configuration: {e}")
        return {
            "error": f"Failed to get browser configuration: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }