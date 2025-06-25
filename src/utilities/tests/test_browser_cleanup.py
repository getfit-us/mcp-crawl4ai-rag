"""Tests for browser cleanup utilities."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from crawl4ai_mcp.utilities.browser_cleanup import (
    BrowserProcessManager,
    find_orphaned_chrome_processes,
    cleanup_orphaned_chrome_processes,
    get_chrome_process_count,
    emergency_browser_cleanup,
    get_browser_manager
)


class TestBrowserProcessManager:
    """Test the BrowserProcessManager class."""
    
    def test_track_process(self):
        """Test process tracking functionality."""
        manager = BrowserProcessManager()
        
        # Track some processes
        manager.track_process(12345)
        manager.track_process(67890)
        
        assert 12345 in manager._tracked_pids
        assert 67890 in manager._tracked_pids
        assert len(manager._tracked_pids) == 2
    
    def test_untrack_process(self):
        """Test process untracking functionality."""
        manager = BrowserProcessManager()
        
        # Track and then untrack
        manager.track_process(12345)
        assert 12345 in manager._tracked_pids
        
        manager.untrack_process(12345)
        assert 12345 not in manager._tracked_pids
        
        # Untracking non-existent process should not error
        manager.untrack_process(99999)
    
    @pytest.mark.asyncio
    async def test_cleanup_tracked_processes_empty(self):
        """Test cleanup with no tracked processes."""
        manager = BrowserProcessManager()
        
        cleaned = await manager.cleanup_tracked_processes()
        assert cleaned == []
    
    @pytest.mark.asyncio
    @patch('psutil.Process')
    async def test_cleanup_tracked_processes_success(self, mock_process_class):
        """Test successful cleanup of tracked processes."""
        manager = BrowserProcessManager()
        
        # Mock process
        mock_process = Mock()
        mock_process.name.return_value = "chrome"
        mock_process.cmdline.return_value = ["chrome", "--headless"]
        mock_process.terminate.return_value = None
        mock_process.wait.return_value = None
        mock_process_class.return_value = mock_process
        
        # Track a process
        manager.track_process(12345)
        
        # Clean up
        cleaned = await manager.cleanup_tracked_processes(timeout=1)
        
        assert cleaned == [12345]
        assert 12345 not in manager._tracked_pids
        mock_process.terminate.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('psutil.Process')
    async def test_cleanup_process_not_browser(self, mock_process_class):
        """Test cleanup skips non-browser processes."""
        manager = BrowserProcessManager()
        
        # Mock non-browser process
        mock_process = Mock()
        mock_process.name.return_value = "python"
        mock_process.cmdline.return_value = ["python", "script.py"]
        mock_process_class.return_value = mock_process
        
        result = await manager._cleanup_process(12345, timeout=1)
        assert result is True
        mock_process.terminate.assert_not_called()


class TestBrowserUtilityFunctions:
    """Test browser utility functions."""
    
    def test_get_chrome_process_count(self):
        """Test Chrome process counting."""
        with patch('psutil.process_iter') as mock_process_iter:
            # Mock processes
            chrome_proc = Mock()
            chrome_proc.info = {'name': 'chrome'}
            
            python_proc = Mock()
            python_proc.info = {'name': 'python'}
            
            chromium_proc = Mock()
            chromium_proc.info = {'name': 'chromium-browser'}
            
            mock_process_iter.return_value = [chrome_proc, python_proc, chromium_proc]
            
            count = get_chrome_process_count()
            assert count == 2  # chrome and chromium-browser
    
    @patch('psutil.process_iter')
    def test_find_orphaned_chrome_processes(self, mock_process_iter):
        """Test finding orphaned Chrome processes."""
        import time
        
        # Mock orphaned Chrome process (old, headless, parent is init)
        orphaned_proc = Mock()
        orphaned_proc.info = {
            'pid': 12345,
            'name': 'chrome',
            'cmdline': ['chrome', '--headless', '--remote-debugging-port=9222'],
            'create_time': time.time() - 400  # 400 seconds old (> 5 minutes)
        }
        
        # Mock parent process (init)
        mock_parent = Mock()
        mock_parent.name.return_value = 'init'
        
        # Mock psutil.Process for the orphaned process
        with patch('psutil.Process') as mock_proc_class:
            mock_process_instance = Mock()
            mock_process_instance.parent.return_value = mock_parent
            mock_proc_class.return_value = mock_process_instance
            
            mock_process_iter.return_value = [orphaned_proc]
            
            orphaned = find_orphaned_chrome_processes()
            assert len(orphaned) == 1
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.utilities.browser_cleanup.find_orphaned_chrome_processes')
    async def test_cleanup_orphaned_chrome_processes_none_found(self, mock_find):
        """Test cleanup when no orphaned processes are found."""
        mock_find.return_value = []
        
        count = await cleanup_orphaned_chrome_processes()
        assert count == 0
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.utilities.browser_cleanup.find_orphaned_chrome_processes')
    async def test_cleanup_orphaned_chrome_processes_success(self, mock_find):
        """Test successful cleanup of orphaned processes."""
        # Mock orphaned process
        mock_proc = Mock()
        mock_proc.pid = 12345
        mock_proc.terminate.return_value = None
        mock_proc.wait.return_value = None
        mock_find.return_value = [mock_proc]
        
        count = await cleanup_orphaned_chrome_processes()
        assert count == 1
        mock_proc.terminate.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('crawl4ai_mcp.utilities.browser_cleanup.get_browser_manager')
    @patch('crawl4ai_mcp.utilities.browser_cleanup.cleanup_orphaned_chrome_processes')
    async def test_emergency_browser_cleanup(self, mock_cleanup_orphaned, mock_get_manager):
        """Test emergency browser cleanup."""
        # Mock manager
        mock_manager = Mock()
        mock_manager.cleanup_tracked_processes = AsyncMock(return_value=[12345])
        mock_get_manager.return_value = mock_manager
        
        # Mock orphaned cleanup
        mock_cleanup_orphaned.return_value = 2
        
        await emergency_browser_cleanup()
        
        mock_manager.cleanup_tracked_processes.assert_called_once()
        mock_cleanup_orphaned.assert_called_once()


class TestBrowserManagerSingleton:
    """Test the global browser manager functionality."""
    
    def test_get_browser_manager_singleton(self):
        """Test that get_browser_manager returns the same instance."""
        manager1 = get_browser_manager()
        manager2 = get_browser_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, BrowserProcessManager)


@pytest.mark.integration
class TestBrowserCleanupIntegration:
    """Integration tests for browser cleanup (requires actual system processes)."""
    
    def test_real_process_count(self):
        """Test getting real Chrome process count."""
        # This test will work regardless of whether Chrome is running
        count = get_chrome_process_count()
        assert isinstance(count, int)
        assert count >= 0
    
    def test_real_orphaned_processes(self):
        """Test finding real orphaned processes."""
        # This test will work regardless of system state
        orphaned = find_orphaned_chrome_processes()
        assert isinstance(orphaned, list)
        
        # All items should be psutil.Process instances
        for proc in orphaned:
            assert hasattr(proc, 'pid')
            assert hasattr(proc, 'name')
    
    @pytest.mark.asyncio
    async def test_manager_with_fake_pid(self):
        """Test manager operations with a fake PID."""
        manager = BrowserProcessManager()
        
        # Track a fake PID (should not exist)
        fake_pid = 999999
        manager.track_process(fake_pid)
        
        # Cleanup should handle non-existent process gracefully
        cleaned = await manager.cleanup_tracked_processes(timeout=1)
        
        # The fake PID should be "cleaned" (removed from tracking)
        # because psutil will raise NoSuchProcess
        assert fake_pid in cleaned or len(cleaned) == 0  # Depends on implementation
        assert fake_pid not in manager._tracked_pids