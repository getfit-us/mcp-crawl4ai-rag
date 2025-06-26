"""Simple Chrome cleanup tool for MCP server."""

import logging
import subprocess
import sys
from typing import Dict, Any
from datetime import datetime

from crawl4ai_mcp.mcp_server import mcp

logger = logging.getLogger(__name__)


@mcp.tool()
def cleanup_chrome_processes() -> Dict[str, Any]:
    """
    Manually clean up Chrome processes that may be left running.
    
    This is a simple tool that kills Chrome processes. Use with caution.
    
    Returns:
        Dict with cleanup results
    """
    try:
        timestamp = datetime.now().isoformat()
        
        if sys.platform == "win32":
            # Windows
            result = subprocess.run(
                ["taskkill", "/f", "/im", "chrome.exe"],
                capture_output=True,
                text=True
            )
            result2 = subprocess.run(
                ["taskkill", "/f", "/im", "chromium.exe"],
                capture_output=True,
                text=True
            )
            
            success = result.returncode == 0 or result2.returncode == 0
            message = f"Chrome cleanup attempted. Chrome: {result.stdout.strip()}, Chromium: {result2.stdout.strip()}"
            
        else:
            # Unix-like systems (macOS, Linux)
            result = subprocess.run(
                ["pkill", "-f", "chrome"],
                capture_output=True,
                text=True
            )
            
            success = result.returncode in [0, 1]  # 0 = success, 1 = no processes found
            if result.returncode == 0:
                message = "Chrome processes terminated successfully"
            elif result.returncode == 1:
                message = "No Chrome processes found to terminate"
            else:
                message = f"Chrome cleanup failed: {result.stderr.strip()}"
        
        return {
            "success": success,
            "message": message,
            "timestamp": timestamp,
            "platform": sys.platform
        }
        
    except Exception as e:
        logger.error(f"Error during Chrome cleanup: {e}")
        return {
            "success": False,
            "error": f"Failed to clean up Chrome processes: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "platform": sys.platform
        }


@mcp.tool()
def check_chrome_processes() -> Dict[str, Any]:
    """
    Check how many Chrome processes are currently running.
    
    Returns:
        Dict with process count information
    """
    try:
        timestamp = datetime.now().isoformat()
        
        if sys.platform == "win32":
            # Windows
            result = subprocess.run(
                ["tasklist", "/fi", "imagename eq chrome.exe"],
                capture_output=True,
                text=True
            )
            count = len([line for line in result.stdout.split('\n') if 'chrome.exe' in line])
            
        else:
            # Unix-like systems
            result = subprocess.run(
                ["pgrep", "-c", "-f", "chrome"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                count = int(result.stdout.strip())
            elif result.returncode == 1:
                count = 0  # No processes found
            else:
                count = -1  # Error occurred
        
        return {
            "chrome_process_count": count,
            "timestamp": timestamp,
            "platform": sys.platform,
            "status": "healthy" if count < 10 else "warning" if count < 20 else "high"
        }
        
    except Exception as e:
        logger.error(f"Error checking Chrome processes: {e}")
        return {
            "error": f"Failed to check Chrome processes: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "platform": sys.platform
        }