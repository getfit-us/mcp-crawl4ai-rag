"""Browser process cleanup and monitoring utilities."""

import logging
import psutil
import time
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class BrowserProcessManager:
    """Manages browser processes and cleanup operations."""
    
    def __init__(self) -> None:
        self._tracked_pids: Set[int] = set()
        self._cleanup_timeout: int = 10
        
    def track_process(self, pid: int) -> None:
        """Track a browser process for cleanup."""
        self._tracked_pids.add(pid)
        logger.debug(f"Tracking browser process PID: {pid}")
    
    def untrack_process(self, pid: int) -> None:
        """Stop tracking a browser process."""
        self._tracked_pids.discard(pid)
        logger.debug(f"Untracked browser process PID: {pid}")
    
    async def cleanup_tracked_processes(self, timeout: int = 10) -> List[int]:
        """
        Clean up all tracked browser processes.
        
        Args:
            timeout: Maximum time to wait for cleanup in seconds
            
        Returns:
            List of PIDs that were successfully cleaned up
        """
        if not self._tracked_pids:
            return []
            
        logger.info(f"Cleaning up {len(self._tracked_pids)} tracked browser processes")
        cleaned_pids = []
        
        for pid in list(self._tracked_pids):
            try:
                if await self._cleanup_process(pid, timeout):
                    cleaned_pids.append(pid)
                    self.untrack_process(pid)
            except Exception as e:
                logger.warning(f"Failed to cleanup process {pid}: {e}")
        
        return cleaned_pids
    
    async def _cleanup_process(self, pid: int, timeout: int) -> bool:
        """
        Clean up a single process gracefully.
        
        Args:
            pid: Process ID to cleanup
            timeout: Maximum time to wait
            
        Returns:
            True if process was cleaned up successfully
        """
        try:
            process = psutil.Process(pid)
            
            # Check if it's actually a Chrome/browser process
            if not self._is_browser_process(process):
                logger.debug(f"PID {pid} is not a browser process, skipping")
                return True
            
            logger.debug(f"Attempting graceful shutdown of browser process {pid}")
            
            # Try graceful termination first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=timeout // 2)
                logger.debug(f"Browser process {pid} terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                logger.warning(f"Browser process {pid} did not terminate gracefully, forcing kill")
                
            # Force kill if graceful termination failed
            process.kill()
            try:
                process.wait(timeout=timeout // 2)
                logger.info(f"Browser process {pid} killed forcefully")
                return True
            except psutil.TimeoutExpired:
                logger.error(f"Failed to kill browser process {pid}")
                return False
                
        except psutil.NoSuchProcess:
            logger.debug(f"Browser process {pid} already terminated")
            return True
        except psutil.AccessDenied:
            logger.warning(f"Access denied when trying to cleanup process {pid}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error cleaning up process {pid}: {e}")
            return False
    
    def _is_browser_process(self, process: psutil.Process) -> bool:
        """Check if a process is a browser process."""
        try:
            name = process.name().lower()
            cmdline = ' '.join(process.cmdline()).lower()
            
            browser_indicators = [
                'chrome', 'chromium', 'google-chrome', 'google-chrome-stable',
                'firefox', 'safari', 'edge', 'msedge', 'opera'
            ]
            
            return any(indicator in name or indicator in cmdline for indicator in browser_indicators)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


def find_orphaned_chrome_processes() -> List[psutil.Process]:
    """
    Find orphaned Chrome processes that might have been left running.
    
    Returns:
        List of Chrome processes that appear to be orphaned
    """
    orphaned = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info['cmdline']).lower() if proc.info['cmdline'] else ''
                
                # Look for Chrome processes
                if ('chrome' in name or 'chromium' in name) and proc.info['pid'] != 0:
                    # Check if it's a headless Chrome process (common for web scraping)
                    if '--headless' in cmdline or '--remote-debugging-port' in cmdline:
                        # Check if process has been running for more than 5 minutes without a parent
                        current_time = time.time()
                        process_age = current_time - proc.info['create_time']
                        
                        if process_age > 300:  # 5 minutes
                            try:
                                parent = psutil.Process(proc.info['pid']).parent()
                                if parent is None or parent.name().lower() in ['init', 'systemd']:
                                    orphaned.append(psutil.Process(proc.info['pid']))
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
    except Exception as e:
        logger.warning(f"Error scanning for orphaned Chrome processes: {e}")
    
    return orphaned


async def cleanup_orphaned_chrome_processes() -> int:
    """
    Clean up orphaned Chrome processes.
    
    Returns:
        Number of processes cleaned up
    """
    orphaned = find_orphaned_chrome_processes()
    
    if not orphaned:
        logger.debug("No orphaned Chrome processes found")
        return 0
    
    logger.info(f"Found {len(orphaned)} orphaned Chrome processes")
    cleaned_count = 0
    
    for proc in orphaned:
        try:
            logger.info(f"Cleaning up orphaned Chrome process PID: {proc.pid}")
            
            # Try graceful termination first
            proc.terminate()
            try:
                proc.wait(timeout=5)
                cleaned_count += 1
                logger.info(f"Orphaned Chrome process {proc.pid} terminated gracefully")
                continue
            except psutil.TimeoutExpired:
                pass
            
            # Force kill if needed
            proc.kill()
            try:
                proc.wait(timeout=5)
                cleaned_count += 1
                logger.info(f"Orphaned Chrome process {proc.pid} killed forcefully")
            except psutil.TimeoutExpired:
                logger.error(f"Failed to kill orphaned Chrome process {proc.pid}")
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Could not cleanup process {proc.pid}: {e}")
            
    return cleaned_count


def get_chrome_process_count() -> int:
    """Get the current number of Chrome processes running."""
    count = 0
    try:
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info['name'].lower()
                if 'chrome' in name or 'chromium' in name:
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.warning(f"Error counting Chrome processes: {e}")
    
    return count


# Global browser process manager instance
_browser_manager: Optional[BrowserProcessManager] = None


def get_browser_manager() -> BrowserProcessManager:
    """Get or create the global browser process manager."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserProcessManager()
    return _browser_manager


async def emergency_browser_cleanup() -> None:
    """Perform emergency cleanup of all browser processes."""
    logger.warning("Performing emergency browser cleanup")
    
    # Clean up tracked processes
    manager = get_browser_manager()
    tracked_cleaned = await manager.cleanup_tracked_processes(timeout=5)
    
    # Clean up orphaned processes
    orphaned_cleaned = await cleanup_orphaned_chrome_processes()
    
    total_cleaned = len(tracked_cleaned) + orphaned_cleaned
    if total_cleaned > 0:
        logger.info(f"Emergency cleanup completed: {total_cleaned} processes cleaned up")
    else:
        logger.info("Emergency cleanup completed: no processes needed cleanup")