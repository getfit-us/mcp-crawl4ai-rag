"""Browser health monitoring and periodic cleanup utilities."""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from crawl4ai_mcp.utilities.browser_cleanup import (
    get_chrome_process_count,
    find_orphaned_chrome_processes,
    cleanup_orphaned_chrome_processes
)

logger = logging.getLogger(__name__)


class BrowserHealthMonitor:
    """Monitors browser health and performs periodic cleanup."""
    
    def __init__(
        self,
        check_interval: int = 300,  # 5 minutes
        max_processes_threshold: int = 20,
        auto_cleanup_orphaned: bool = True,
        max_orphaned_age_minutes: int = 10
    ):
        self.check_interval = check_interval
        self.max_processes_threshold = max_processes_threshold
        self.auto_cleanup_orphaned = auto_cleanup_orphaned
        self.max_orphaned_age_minutes = max_orphaned_age_minutes
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_check = datetime.now()
        self._total_cleanups = 0
        
    async def start_monitoring(self) -> None:
        """Start the browser health monitoring."""
        if self._monitoring:
            logger.warning("Browser monitoring is already running")
            return
            
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Browser health monitoring started (interval: {self.check_interval}s)")
        
    async def stop_monitoring(self) -> None:
        """Stop the browser health monitoring."""
        if not self._monitoring:
            return
            
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            
        logger.info(f"Browser health monitoring stopped (total cleanups: {self._total_cleanups})")
        
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self._monitoring:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.debug("Browser monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in browser monitoring loop: {e}")
            
    async def _perform_health_check(self) -> None:
        """Perform a single health check and cleanup if needed."""
        try:
            self._last_check = datetime.now()
            
            # Count total Chrome processes
            total_count = get_chrome_process_count()
            
            # Find orphaned processes
            orphaned = find_orphaned_chrome_processes()
            orphaned_count = len(orphaned)
            
            logger.debug(f"Browser health check: {total_count} total, {orphaned_count} orphaned")
            
            # Check for too many processes
            if total_count > self.max_processes_threshold:
                logger.warning(
                    f"High Chrome process count detected: {total_count} "
                    f"(threshold: {self.max_processes_threshold})"
                )
                
            # Auto-cleanup orphaned processes if enabled
            if self.auto_cleanup_orphaned and orphaned_count > 0:
                logger.info(f"Auto-cleaning {orphaned_count} orphaned Chrome processes")
                
                cleaned_count = await cleanup_orphaned_chrome_processes()
                if cleaned_count > 0:
                    self._total_cleanups += cleaned_count
                    logger.info(f"Successfully cleaned up {cleaned_count} orphaned processes")
                else:
                    logger.warning("No orphaned processes were cleaned up")
                    
        except Exception as e:
            logger.error(f"Error during browser health check: {e}")
            
    async def get_status(self) -> dict:
        """Get current monitoring status."""
        return {
            "monitoring": self._monitoring,
            "last_check": self._last_check.isoformat(),
            "check_interval": self.check_interval,
            "total_cleanups": self._total_cleanups,
            "current_chrome_processes": get_chrome_process_count(),
            "current_orphaned_processes": len(find_orphaned_chrome_processes())
        }
        
    async def force_cleanup(self) -> int:
        """Force an immediate cleanup of orphaned processes."""
        logger.info("Force cleanup requested")
        orphaned_count = len(find_orphaned_chrome_processes())
        
        if orphaned_count == 0:
            logger.info("No orphaned processes found for force cleanup")
            return 0
            
        cleaned_count = await cleanup_orphaned_chrome_processes()
        if cleaned_count > 0:
            self._total_cleanups += cleaned_count
            logger.info(f"Force cleanup completed: {cleaned_count} processes cleaned")
        
        return cleaned_count


# Global monitor instance
_browser_monitor: Optional[BrowserHealthMonitor] = None


def get_browser_monitor() -> BrowserHealthMonitor:
    """Get or create the global browser health monitor."""
    global _browser_monitor
    if _browser_monitor is None:
        _browser_monitor = BrowserHealthMonitor()
    return _browser_monitor


async def start_browser_monitoring(
    check_interval: int = 300,
    max_processes_threshold: int = 20,
    auto_cleanup_orphaned: bool = True
) -> None:
    """Start browser health monitoring with specified settings."""
    monitor = get_browser_monitor()
    monitor.check_interval = check_interval
    monitor.max_processes_threshold = max_processes_threshold
    monitor.auto_cleanup_orphaned = auto_cleanup_orphaned
    
    await monitor.start_monitoring()


async def stop_browser_monitoring() -> None:
    """Stop browser health monitoring."""
    monitor = get_browser_monitor()
    await monitor.stop_monitoring()


async def get_browser_monitoring_status() -> dict:
    """Get current browser monitoring status."""
    monitor = get_browser_monitor()
    return await monitor.get_status()