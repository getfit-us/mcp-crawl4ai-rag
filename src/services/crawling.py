"""Crawling service for web content extraction and processing."""

import asyncio
import json
import logging
import requests
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

# Constants
MAX_FAILURES_TO_LOG = 5


class CrawlCancelledException(Exception):
    """Exception raised when a crawl operation is cancelled."""
    pass


class CrawlingService:
    """Service for crawling web content and extracting information."""
    
    def __init__(self, crawler: AsyncWebCrawler, settings: Optional[Any] = None, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize crawling service.
        
        Args:
            crawler: AsyncWebCrawler instance
            settings: Application settings (optional)
            embedding_service: Embedding service instance (optional)
        """
        self.crawler = crawler
        self.settings = settings or get_settings()
        self.embedding_service =  EmbeddingService(self.settings)
        
        # Cancellation tracking
        self._active_crawls: Dict[str, Dict[str, Any]] = {}
        self._cancellation_events: Dict[str, asyncio.Event] = {}
        self._crawl_lock = asyncio.Lock()
    
    async def start_crawl_operation(self, operation_type: str, urls: List[str]) -> str:
        """
        Start tracking a new crawl operation.
        
        Args:
            operation_type: Type of crawl operation (e.g., "batch", "recursive", "sitemap")
            urls: List of URLs being crawled
            
        Returns:
            Unique crawl ID for this operation
        """
        crawl_id = str(uuid.uuid4())
        async with self._crawl_lock:
            self._active_crawls[crawl_id] = {
                "operation_type": operation_type,
                "urls": urls,
                "start_time": asyncio.get_event_loop().time(),
                "status": "running"
            }
            self._cancellation_events[crawl_id] = asyncio.Event()
        
        logger.info(f"Started crawl operation {crawl_id} of type {operation_type} with {len(urls)} URLs")
        return crawl_id
    
    async def finish_crawl_operation(self, crawl_id: str, status: str = "completed"):
        """
        Mark a crawl operation as finished.
        
        Args:
            crawl_id: ID of the crawl operation
            status: Final status of the operation
        """
        async with self._crawl_lock:
            if crawl_id in self._active_crawls:
                self._active_crawls[crawl_id]["status"] = status
                # Keep the record for a short time for status queries
                await asyncio.sleep(0.1)  # Allow any pending status checks
                del self._active_crawls[crawl_id]
                del self._cancellation_events[crawl_id]
        
        logger.info(f"Finished crawl operation {crawl_id} with status {status}")
    
    async def cancel_crawl_operation(self, crawl_id: str) -> bool:
        """
        Cancel a specific crawl operation.
        
        Args:
            crawl_id: ID of the crawl operation to cancel
            
        Returns:
            True if the operation was cancelled, False if not found
        """
        async with self._crawl_lock:
            if crawl_id in self._cancellation_events:
                self._cancellation_events[crawl_id].set()
                if crawl_id in self._active_crawls:
                    self._active_crawls[crawl_id]["status"] = "cancelled"
                logger.info(f"Cancelled crawl operation {crawl_id}")
                return True
            return False
    
    async def cancel_all_crawl_operations(self) -> int:
        """
        Cancel all active crawl operations.
        
        Returns:
            Number of operations cancelled
        """
        cancelled_count = 0
        async with self._crawl_lock:
            for crawl_id in list(self._cancellation_events.keys()):
                self._cancellation_events[crawl_id].set()
                if crawl_id in self._active_crawls:
                    self._active_crawls[crawl_id]["status"] = "cancelled"
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} crawl operations")
        return cancelled_count
    
    async def get_active_crawl_operations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active crawl operations.
        
        Returns:
            Dictionary mapping crawl IDs to operation info
        """
        async with self._crawl_lock:
            return self._active_crawls.copy()
    
    def _check_cancellation(self, crawl_id: str):
        """
        Check if a crawl operation should be cancelled.
        
        Args:
            crawl_id: ID of the crawl operation
            
        Raises:
            CrawlCancelledException: If the operation has been cancelled
        """
        if crawl_id in self._cancellation_events and self._cancellation_events[crawl_id].is_set():
            raise CrawlCancelledException(f"Crawl operation {crawl_id} was cancelled")
    
    def is_sitemap(self, url: str) -> bool:
        """
        Check if a URL is a sitemap.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is a sitemap, False otherwise
        """
        return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path
    
    def is_txt(self, url: str) -> bool:
        """
        Check if a URL is a text file.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is a text file, False otherwise
        """
        return url.endswith('.txt')
    
    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap and extract URLs.
        
        Args:
            sitemap_url: URL of the sitemap
            
        Returns:
            List of URLs found in the sitemap
        """
        resp = requests.get(sitemap_url)
        urls = []

        if resp.status_code == 200:
            try:
                tree = ElementTree.fromstring(resp.content)
                urls = [loc.text for loc in tree.findall('.//{*}loc')]
            except Exception as e:
                logger.error(f"Error parsing sitemap XML: {e}")

        return urls
    
    def remove_think_tags(self, text: str) -> str:
        """
        Remove <think> tags from text.
        """
        return text.replace("<think>", "").replace("</think>", "")
    
    async def crawl_markdown_file(self, url: str, crawl_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Crawl a .txt or markdown file.
        
        Args:
            url: URL of the file
            crawl_id: Optional crawl ID for cancellation tracking
            
        Returns:
            List of dictionaries with URL and markdown content
        """
        if crawl_id:
            self._check_cancellation(crawl_id)
            
        crawl_config = CrawlerRunConfig()

        result = await self.crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{'url': url, 'markdown': result.markdown}]
        else:
            logger.warning(f"Failed to crawl {url}: {result.error_message}")
            return []
    
    async def crawl_batch(self, urls: List[str], max_concurrent: int = 10, crawl_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Batch crawl multiple URLs in parallel.
        
        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum number of concurrent browser sessions
            crawl_id: Optional crawl ID for cancellation tracking
            
        Returns:
            List of dictionaries with URL and markdown content
        """
        if crawl_id:
            self._check_cancellation(crawl_id)
            
        logger.debug(f"Starting crawl_batch with {len(urls)} URLs")
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent
        )

        try:
            # Check for cancellation before starting
            if crawl_id:
                self._check_cancellation(crawl_id)
                
            results = await self.crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
            logger.debug(f"arun_many returned {len(results)} results")
            
            # Check for cancellation after crawling
            if crawl_id:
                self._check_cancellation(crawl_id)
            
            successful_results = [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
            logger.info(f"Filtered to {len(successful_results)} successful results from {len(urls)} URLs")
            
            # Log some failures if any
            failed_results = [r for r in results if not r.success or not r.markdown]
            if failed_results:
                logger.error(f"{len(failed_results)} URLs failed to crawl")
                for failed in failed_results[:MAX_FAILURES_TO_LOG]:
                    logger.error(f"Failed: {failed.url} - Error: {getattr(failed, 'error_message', 'No markdown content')}")
            
            return successful_results
        except Exception as e:
            logger.error(f"Error in crawl_batch: {e}")
            return []
    
    async def crawl_recursive_internal_links(
        self, 
        start_urls: List[str], 
        max_depth: int = 3, 
        max_concurrent: int = 10,
        crawl_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively crawl internal links from start URLs up to a maximum depth.
        
        Args:
            start_urls: List of starting URLs
            max_depth: Maximum recursion depth
            max_concurrent: Maximum number of concurrent browser sessions
            crawl_id: Optional crawl ID for cancellation tracking
            
        Returns:
            List of dictionaries with URL and markdown content
        """
        if crawl_id:
            self._check_cancellation(crawl_id)
            
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent
        )

        visited = set()

        def normalize_url(url):
            return urldefrag(url)[0]

        current_urls = set([normalize_url(u) for u in start_urls])
        results_all = []

        for depth in range(max_depth):
            # Check for cancellation at the start of each depth level
            if crawl_id:
                self._check_cancellation(crawl_id)
                
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
            if not urls_to_crawl:
                break

            results = await self.crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
            next_level_urls = set()

            for result in results:
                # Check for cancellation periodically during processing
                if crawl_id:
                    self._check_cancellation(crawl_id)
                    
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({'url': result.url, 'markdown': result.markdown})
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)

            current_urls = next_level_urls

        return results_all
    
    def extract_code_blocks(self, markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
        """
        Extract code blocks from markdown content along with context.
        
        Args:
            markdown_content: The markdown content to extract code blocks from
            min_length: Minimum length of code blocks to extract (default: 1000 characters)
            
        Returns:
            List of dictionaries containing code blocks and their context
        """
        code_blocks = []
        
        # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
        content = markdown_content.strip()
        start_offset = 0
        if content.startswith('```'):
            # Skip the first triple backticks
            start_offset = 3
            logger.debug("Skipping initial triple backticks")
        
        # Find all occurrences of triple backticks
        backtick_positions = []
        pos = start_offset
        while True:
            pos = markdown_content.find('```', pos)
            if pos == -1:
                break
            backtick_positions.append(pos)
            pos += 3
        
        # Process pairs of backticks
        i = 0
        while i < len(backtick_positions) - 1:
            start_pos = backtick_positions[i]
            end_pos = backtick_positions[i + 1]
            
            # Extract the content between backticks
            code_section = markdown_content[start_pos+3:end_pos]
            
            # Check if there's a language specifier on the first line
            lines = code_section.split('\n', 1)
            if len(lines) > 1:
                # Check if first line is a language specifier (no spaces, common language names)
                first_line = lines[0].strip()
                if first_line and ' ' not in first_line and len(first_line) < 20:
                    language = first_line
                    code_content = lines[1].strip() if len(lines) > 1 else ""
                else:
                    language = ""
                    code_content = code_section.strip()
            else:
                language = ""
                code_content = code_section.strip()
            
            # Skip if code block is too short
            if len(code_content) < min_length:
                i += 2  # Move to next pair
                continue
            
            # Extract context before (1000 chars)
            context_start = max(0, start_pos - 1000)
            context_before = markdown_content[context_start:start_pos].strip()
            
            # Extract context after (1000 chars)
            context_end = min(len(markdown_content), end_pos + 3 + 1000)
            context_after = markdown_content[end_pos + 3:context_end].strip()
            
            code_blocks.append({
                'code': code_content,
                'language': language,
                'context_before': context_before,
                'context_after': context_after,
                'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
            })
            
            # Move to next pair (skip the closing backtick we just processed)
            i += 2
        
        return code_blocks
    
    async def generate_code_example_summary(
        self, 
        code: str, 
        context_before: str, 
        context_after: str
    ) -> str:
        """
        Generate a summary for a code example using its surrounding context.
        
        Args:
            code: The code example
            context_before: Context before the code
            context_after: Context after the code
            
        Returns:
            A summary of what the code example demonstrates
        """
        # Create the prompt
        prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
        
        try:
            # Use embedding service's OpenAI client
            response = await self.embedding_service._run_in_executor(
                lambda: self.embedding_service.client.chat.completions.create(
                    model=self.settings.summary_llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
            )
            
           
            
            return self.remove_think_tags(response.choices[0].message.content.strip() if response.choices[0].message.content else "")
        
        except Exception as e:
            logger.error(f"Error generating code example summary: {e}")
            return "Code example for demonstration purposes."
    
    async def extract_source_summary(
        self, 
        source_id: str, 
        content: str, 
        max_length: int = 500
    ) -> str:
        """
        Extract a summary for a source from its content using an LLM.
        
        This function uses the OpenAI API to generate a concise summary of the source content.
        
        Args:
            source_id: The source ID (domain)
            content: The content to extract a summary from
            max_length: Maximum length of the summary
            
        Returns:
            A summary string
        """
        # Default summary if we can't extract anything meaningful
        default_summary = f"Content from {source_id}"
        
        if not content or len(content.strip()) == 0:
            return default_summary
        
        # Limit content length to avoid token limits
        truncated_content = content[:25000] if len(content) > 25000 else content
        
        # Create the prompt for generating the summary
        prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
        
        try:
            # Use embedding service's OpenAI client
            response = await self.embedding_service._run_in_executor(
                lambda: self.embedding_service.client.chat.completions.create(
                    model=self.settings.summary_llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
            )
           
            summary_content = response.choices[0].message.content
            if not summary_content:
                logger.warning(f"LLM returned empty content for {source_id}. Using default summary.")
                return default_summary

            summary = self.remove_think_tags(summary_content.strip())
            
            if not summary:
                logger.warning(f"LLM returned empty summary for {source_id}. Using default summary.")
                
            # Ensure the summary is not too long
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
            return default_summary

    async def generate_code_example_summaries_batch(
        self,
        code_examples_data: List[Tuple[str, str, str]]
    ) -> List[str]:
        """
        Generate summaries for multiple code examples in batch.
        
        Args:
            code_examples_data: List of tuples containing (code, context_before, context_after)
            
        Returns:
            List of summary strings
        """
        if not code_examples_data:
            return []
        
        # Prepare batch prompts
        batch_prompts = []
        for code, context_before, context_after in code_examples_data:
            prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
            batch_prompts.append(prompt)
        
        try:
            # Process batch requests in parallel
            tasks = []
            for prompt in batch_prompts:
                task = self.embedding_service._run_in_executor(
                    lambda p=prompt: self.embedding_service.client.chat.completions.create(
                        model=self.settings.summary_llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                            {"role": "user", "content": p}
                        ],
                        temperature=0.3,
                        max_tokens=4000
                    )
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            summaries = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Error generating code example summary {i}: {response}")
                    summaries.append("Code example for demonstration purposes.")
                else:
                    try:
                        content = response.choices[0].message.content
                        summary = self.remove_think_tags(content.strip() if content else "")
                        summaries.append(summary if summary else "Code example for demonstration purposes.")
                    except Exception as e:
                        logger.error(f"Error processing code example summary response {i}: {e}")
                        summaries.append("Code example for demonstration purposes.")
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error in batch code example summary generation: {e}")
            # Return default summaries
            return ["Code example for demonstration purposes." for _ in code_examples_data]

    async def extract_source_summaries_batch(
        self,
        source_data: List[Tuple[str, str]]
    ) -> List[str]:
        """
        Extract summaries for multiple sources in batch.
        
        Args:
            source_data: List of tuples containing (source_id, content)
            
        Returns:
            List of summary strings
        """
        if not source_data:
            return []
        
        # Prepare batch prompts
        batch_prompts = []
        default_summaries = []
        for source_id, content in source_data:
            # Default summary if we can't extract anything meaningful
            default_summary = f"Content from {source_id}"
            default_summaries.append(default_summary)
            
            if not content or len(content.strip()) == 0:
                batch_prompts.append(None)  # Skip empty content
                continue
            
            # Limit content length to avoid token limits
            truncated_content = content[:25000] if len(content) > 25000 else content
            
            prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
            batch_prompts.append(prompt)
        
        try:
            # Process batch requests in parallel
            tasks = []
            for i, prompt in enumerate(batch_prompts):
                if prompt is None:
                    tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task for empty content
                else:
                    task = self.embedding_service._run_in_executor(
                        lambda p=prompt: self.embedding_service.client.chat.completions.create(
                            model=self.settings.summary_llm_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                                {"role": "user", "content": p}
                            ],
                            temperature=0.3,
                            max_tokens=4000
                        )
                    )
                    tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            summaries = []
            for i, (source_id, content) in enumerate(source_data):
                if batch_prompts[i] is None:
                    # Empty content, use default
                    summaries.append(default_summaries[i])
                elif isinstance(responses[i], Exception):
                    logger.error(f"Error generating source summary for {source_id}: {responses[i]}")
                    summaries.append(default_summaries[i])
                else:
                    try:
                        response_content = responses[i].choices[0].message.content
                        if not response_content:
                            logger.warning(f"LLM returned empty content for {source_id}. Using default summary.")
                            summaries.append(default_summaries[i])
                            continue
                        
                        summary = self.remove_think_tags(response_content.strip())
                        
                        if not summary:
                            logger.warning(f"LLM returned empty summary for {source_id}. Using default summary.")
                            summaries.append(default_summaries[i])
                            continue
                        
                        # Ensure the summary is not too long
                        max_length = 500
                        if len(summary) > max_length:
                            summary = summary[:max_length] + "..."
                        
                        summaries.append(summary)
                        
                    except Exception as e:
                        logger.error(f"Error processing source summary response for {source_id}: {e}")
                        summaries.append(default_summaries[i])
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error in batch source summary generation: {e}")
            # Return default summaries
            return default_summaries