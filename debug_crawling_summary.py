import asyncio
import logging
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler
from crawl4ai_mcp.services.crawling import CrawlingService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

async def main():
    """Main function to run the crawling service test."""
    logger.info("Initializing CrawlingService for debugging...")
    
    # Initialize the web crawler
    crawler = AsyncWebCrawler()
    
    # Initialize CrawlingService
    crawling_service = CrawlingService(crawler=crawler)
    
    # Sample data for testing
    code_example = """
def hello_world():
    print("Hello, World!")
"""
    context_before = "Here is a simple Python function."
    context_after = "This function prints a greeting."
    
    logger.info("Generating code example summary...")
    
    # Call the function to be tested
    summary = await crawling_service.generate_code_example_summary(
        code=code_example,
        context_before=context_before,
        context_after=context_after
    )
    
    logger.info(f"Generated Summary: {summary}")

if __name__ == "__main__":
    asyncio.run(main()) 