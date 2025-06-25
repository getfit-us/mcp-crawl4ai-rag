#!/usr/bin/env python3
"""
Browser maintenance utility for cleaning up leftover Chrome processes.
This script can be run manually or as a cron job for system maintenance.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utilities.browser_cleanup import (
    find_orphaned_chrome_processes,
    cleanup_orphaned_chrome_processes,
    get_chrome_process_count,
    emergency_browser_cleanup
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def status_command():
    """Show current browser process status."""
    print("Chrome Browser Process Status")
    print("=" * 40)
    
    # Count total Chrome processes
    total_count = get_chrome_process_count()
    print(f"Total Chrome processes: {total_count}")
    
    # Find orphaned processes
    orphaned = find_orphaned_chrome_processes()
    print(f"Orphaned Chrome processes: {len(orphaned)}")
    
    if orphaned:
        print("\nOrphaned processes:")
        for proc in orphaned:
            try:
                print(f"  PID: {proc.pid}, Created: {proc.create_time():.2f}, CMD: {' '.join(proc.cmdline()[:3])}")
            except Exception as e:
                print(f"  PID: {proc.pid}, Error getting details: {e}")
    
    print()


async def cleanup_command(force: bool = False):
    """Clean up orphaned Chrome processes."""
    print("Chrome Browser Cleanup")
    print("=" * 30)
    
    # Show status first
    orphaned = find_orphaned_chrome_processes()
    if not orphaned:
        print("✅ No orphaned Chrome processes found.")
        return
    
    print(f"Found {len(orphaned)} orphaned Chrome processes.")
    
    if not force:
        confirm = input("Do you want to clean them up? (y/N): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return
    
    print("Cleaning up orphaned processes...")
    cleaned_count = await cleanup_orphaned_chrome_processes()
    
    if cleaned_count > 0:
        print(f"✅ Successfully cleaned up {cleaned_count} processes.")
    else:
        print("❌ No processes were cleaned up.")


async def emergency_command():
    """Perform emergency cleanup of all browser processes."""
    print("Emergency Browser Cleanup")
    print("=" * 35)
    print("⚠️  This will attempt to terminate ALL browser-related processes!")
    
    confirm = input("Are you sure you want to continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Emergency cleanup cancelled.")
        return
    
    print("Performing emergency cleanup...")
    await emergency_browser_cleanup()
    print("✅ Emergency cleanup completed.")


async def monitor_command(interval: int = 60, max_processes: int = 20):
    """
    Monitor browser processes continuously.
    
    Args:
        interval: Check interval in seconds
        max_processes: Alert threshold for total processes
    """
    print(f"Browser Process Monitor (checking every {interval}s)")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            total_count = get_chrome_process_count()
            orphaned = find_orphaned_chrome_processes()
            
            timestamp = asyncio.get_event_loop().time()
            print(f"[{timestamp:.0f}] Total: {total_count}, Orphaned: {len(orphaned)}")
            
            # Alert if too many processes
            if total_count > max_processes:
                logger.warning(f"High Chrome process count detected: {total_count} (threshold: {max_processes})")
            
            # Auto-cleanup orphaned processes if found
            if orphaned:
                logger.info(f"Auto-cleaning {len(orphaned)} orphaned processes")
                cleaned = await cleanup_orphaned_chrome_processes()
                print(f"  Auto-cleaned {cleaned} orphaned processes")
            
            await asyncio.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Browser maintenance utility for Chrome process management"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show browser process status')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up orphaned browser processes')
    cleanup_parser.add_argument('--force', action='store_true', 
                               help='Skip confirmation prompt')
    
    # Emergency command
    emergency_parser = subparsers.add_parser('emergency', 
                                           help='Emergency cleanup of all browser processes')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', 
                                         help='Monitor browser processes continuously')
    monitor_parser.add_argument('--interval', type=int, default=60,
                               help='Check interval in seconds (default: 60)')
    monitor_parser.add_argument('--max-processes', type=int, default=20,
                               help='Alert threshold for total processes (default: 20)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == 'status':
        asyncio.run(status_command())
    elif args.command == 'cleanup':
        asyncio.run(cleanup_command(force=args.force))
    elif args.command == 'emergency':
        asyncio.run(emergency_command())
    elif args.command == 'monitor':
        asyncio.run(monitor_command(interval=args.interval, max_processes=args.max_processes))


if __name__ == "__main__":
    main()