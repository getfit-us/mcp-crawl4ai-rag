"""Tool for smart crawling with URL type detection."""

import json
import logging
from urllib.parse import urlparse
import asyncio

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.services.crawling import CrawlingService, CrawlCancelledException
from crawl4ai_mcp.utilities.metadata import create_chunk_metadata
from crawl4ai_mcp.utilities.text_processing import TextProcessor
from crawl4ai_mcp.models import CrawlContext

logger = logging.getLogger(__name__)


@mcp.tool()
async def smart_crawl_url(
    ctx: Context, 
    url: str, 
    max_depth: int = 3, 
    max_concurrent: int = 10,
    chunk_size: int = 5000
) -> str:
    """
    Intelligently crawl websites based on URL type and store content in PostgreSQL.
    
    This tool can handle:
    - Single web pages (crawled with depth)
    - Sitemaps (XML files containing multiple URLs)
    - Text files with URLs (like llms-full.txt)
    - Automatic detection and appropriate crawling strategy
    
    The crawled content is automatically chunked and stored in PostgreSQL with embeddings.
    
    Args:
        ctx: The MCP server provided context
        url: Starting URL to crawl (can be a webpage, sitemap, or text file)
        max_depth: Maximum crawling depth for recursive crawling (1-10)
        max_concurrent: Maximum number of concurrent requests (1-50)
        chunk_size: Maximum size of each content chunk (default: 5000)
    
    Returns:
        Summary of the crawling operation including pages crawled and chunks stored
    """
    try:
        # Get context and services
        context: CrawlContext = ctx.request_context.lifespan_context
        crawler = context.crawler
        postgres_pool = context.supabase_client  # Field name kept for compatibility
        settings = context.settings
        
        # Initialize services
        embedding_service = EmbeddingService(settings)
        database_service = DatabaseService(postgres_pool, settings)
        crawling_service = CrawlingService(
            crawler=context.crawler, 
            settings=settings, 
            embedding_service=embedding_service
        )
        text_processor = TextProcessor(settings, embedding_service)
        
        # Override chunk size if specified
        if chunk_size != 5000:
            actual_chunk_size = chunk_size
        else:
            actual_chunk_size = settings.default_chunk_size
        
        # Determine URL type and crawl accordingly
        all_results = []
        crawl_id = None
        
        try:
            if crawling_service.is_txt(url):
                # Crawl as markdown file
                crawl_id = await crawling_service.start_crawl_operation("txt_file", [url])
                all_results = await crawling_service.crawl_markdown_file(url, crawl_id)
                crawl_type = "txt file"
                
            elif crawling_service.is_sitemap(url):
                # Parse sitemap and crawl all URLs
                logger.info(f"Detected sitemap URL: {url}")
                urls_from_sitemap = crawling_service.parse_sitemap(url)
                logger.info(f"Found {len(urls_from_sitemap)} URLs in sitemap")
                
                if urls_from_sitemap:
                    crawl_id = await crawling_service.start_crawl_operation("sitemap", urls_from_sitemap)
                    logger.info(f"Starting batch crawl of {len(urls_from_sitemap)} URLs")
                    all_results = await crawling_service.crawl_batch(
                        urls_from_sitemap, 
                        max_concurrent=max_concurrent,
                        crawl_id=crawl_id
                    )
                    logger.info(f"Batch crawl completed. Got {len(all_results)} results")
                else:
                    logger.warning("No URLs found in sitemap")
                crawl_type = "sitemap"
                
            else:
                # Crawl recursively following internal links
                crawl_id = await crawling_service.start_crawl_operation("recursive", [url])
                all_results = await crawling_service.crawl_recursive_internal_links(
                    [url], 
                    max_depth=max_depth, 
                    max_concurrent=max_concurrent,
                    crawl_id=crawl_id
                )
                crawl_type = "recursive"
        
        except CrawlCancelledException as cancel_error:
            # Handle crawl cancellation specifically
            if crawl_id:
                await crawling_service.finish_crawl_operation(crawl_id, "cancelled")
            
            return json.dumps({
                "success": False,
                "error": "Crawl operation was cancelled",
                "crawl_id": crawl_id,
                "message": str(cancel_error)
            })
        
        except Exception as crawl_error:
            # Handle other errors
            if crawl_id:
                await crawling_service.finish_crawl_operation(crawl_id, "failed")
            raise crawl_error
        
        finally:
            # Mark operation as completed if it was started
            if crawl_id:
                await crawling_service.finish_crawl_operation(crawl_id, "completed")
        
        if not all_results:
            return json.dumps({
                "success": False,
                "error": "No content was successfully crawled",
                "crawl_type": crawl_type
            })
        
        # Process and store all results
        total_chunks_created = 0
        total_code_examples = 0
        processed_urls = []
        source_ids = set()
        
        # First, collect all unique source IDs
        for result in all_results:
            source_id = urlparse(result['url']).netloc
            source_ids.add(source_id)
        
        # Update source info for each unique source BEFORE processing documents
        if settings.enable_batch_summaries and len(source_ids) > 1:
            logger.info(f"Using batch summaries for {len(source_ids)} sources")
            # Prepare source data for batch processing
            source_data = []
            source_word_counts = []
            
            for source_id in source_ids:
                # Get all content for this source
                source_content = "\n\n".join([
                    r['markdown'] for r in all_results 
                    if urlparse(r['url']).netloc == source_id
                ])
                
                # Calculate total word count
                total_word_count = len(source_content.split())
                source_word_counts.append(total_word_count)
                source_data.append((source_id, source_content[:10000]))
            
            # Generate summaries in batch
            source_summaries = await crawling_service.extract_source_summaries_batch(source_data)
            
            # Update source info
            for i, source_id in enumerate(source_ids):
                await database_service.update_source_info(
                    source_id=source_id,
                    summary=source_summaries[i],
                    word_count=source_word_counts[i]
                )
        else:
            # Process source summaries individually
            for source_id in source_ids:
                # Get all content for this source
                source_content = "\n\n".join([
                    r['markdown'] for r in all_results 
                    if urlparse(r['url']).netloc == source_id
                ])
                
                # Calculate total word count
                total_word_count = len(source_content.split())
                
                # Generate summary
                source_summary = await crawling_service.extract_source_summary(
                    source_id, source_content[:10000]
                )
                
                await database_service.update_source_info(
                    source_id=source_id,
                    summary=source_summary,
                    word_count=total_word_count
                )
        
        async def process_single_result(result):
            """Process a single crawl result with all its chunks and code examples."""
            try:
                result_url = result['url']
                markdown_content = result['markdown']
                
                # Extract source_id
                source_id = urlparse(result_url).netloc
                
                # Chunk the content
                chunks = text_processor.smart_chunk_markdown(
                    markdown_content, 
                    chunk_size=actual_chunk_size
                )
                
                if not chunks:
                    return {
                        'url': result_url,
                        'chunks_created': 0,
                        'code_examples': 0,
                        'success': True
                    }
                
                # Prepare data for storage
                urls = []
                chunk_numbers = []
                contents = []
                embeddings = []
                metadatas = []
                url_to_full_document = {result_url: markdown_content}
                
                # Prepare chunk data
                for i, chunk in enumerate(chunks):
                    urls.append(result_url)
                    chunk_numbers.append(i + 1)
                    contents.append(chunk)
                    
                    # Extract metadata
                    section_info = text_processor.extract_section_info(chunk)
                    metadata = create_chunk_metadata(
                        chunk=chunk,
                        source_id=source_id,
                        url=result_url,
                        chunk_index=i,
                        crawl_type=crawl_type,
                        section_info=section_info
                    )
                    metadatas.append(metadata)
                
                # Generate embeddings (batch or individual)
                if settings.use_contextual_embeddings:
                    if settings.enable_batch_contextual_embeddings and len(chunks) > 1:
                        logger.info(f"Generating contextual embeddings for {len(chunks)} chunks from {result_url} in single batch")
                        # Prepare chunks with documents for batch processing
                        chunks_with_docs = [(chunk, markdown_content) for chunk in chunks]
                        
                        # Process all contextual embeddings in a single batch call for better parallelization
                        contextual_results = await embedding_service.generate_contextual_embeddings_batch(chunks_with_docs)
                        
                        # Get embeddings for contextual content
                        contextual_texts = [result[0] for result in contextual_results]
                        if settings.enable_batch_embeddings:
                            # Process all embeddings in a single batch call for better parallelization
                            logger.info(f"Creating embeddings for {len(contextual_texts)} contextual texts in single batch")
                            embeddings = await embedding_service.create_embeddings_batch(contextual_texts)
                        else:
                            for contextual_text in contextual_texts:
                                embedding = await embedding_service.create_embedding(contextual_text)
                                embeddings.append(embedding)
                    else:
                        # Process individually
                        for chunk in chunks:
                            contextual_content, _ = await text_processor.generate_contextual_embedding(
                                markdown_content, chunk
                            )
                            embedding = await embedding_service.create_embedding(contextual_content)
                            embeddings.append(embedding)
                else:
                    if settings.enable_batch_embeddings and len(chunks) > 1:
                        logger.info(f"Creating embeddings for {len(chunks)} chunks from {result_url} in single batch")
                        # Process all embeddings in a single batch call for better parallelization
                        embeddings = await embedding_service.create_embeddings_batch(chunks)
                    else:
                        # Process individually
                        for chunk in chunks:
                            embedding = await embedding_service.create_embedding(chunk)
                            embeddings.append(embedding)
                
                # Add documents to database
                doc_result = await database_service.add_documents(
                    urls=urls,
                    chunk_numbers=chunk_numbers,
                    contents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    url_to_full_document=url_to_full_document
                )
                
                chunks_created = doc_result.get('count', 0)
                code_examples_count = 0
                
                # Extract and store code examples if enabled
                if settings.use_agentic_rag:
                    code_blocks = crawling_service.extract_code_blocks(markdown_content, min_length=50)
                    
                    if code_blocks:
                        # Process code examples
                        code_urls = []
                        code_chunk_numbers = []
                        code_examples = []
                        code_summaries = []
                        code_embeddings = []
                        code_metadatas = []
                        
                        # Prepare formatted code examples
                        for i, code_block in enumerate(code_blocks):
                            code_urls.append(result_url)
                            code_chunk_numbers.append(i + 1)
                            
                            # Format code with language
                            if code_block['language']:
                                formatted_code = f"```{code_block['language']}\n{code_block['code']}\n```"
                            else:
                                formatted_code = f"```\n{code_block['code']}\n```"
                            
                            code_examples.append(formatted_code)
                        
                        # Generate summaries (batch or individual)
                        if settings.enable_batch_summaries and len(code_blocks) > 1:
                            logger.info(f"Generating summaries for {len(code_blocks)} code examples from {result_url} in single batch")
                            # Prepare data for batch processing
                            code_examples_data = [
                                (code_block['code'], code_block['context_before'], code_block['context_after'])
                                for code_block in code_blocks
                            ]
                            
                            # Process all summaries in a single batch call for better parallelization
                            code_summaries = await crawling_service.generate_code_example_summaries_batch(code_examples_data)
                        else:
                            # Process summaries individually
                            for code_block in code_blocks:
                                summary = await crawling_service.generate_code_example_summary(
                                    code_block['code'],
                                    code_block['context_before'],
                                    code_block['context_after']
                                )
                                code_summaries.append(summary)
                        
                        # Generate embeddings for code examples
                        embedding_texts = [f"{summary}\n\n{formatted_code}" 
                                         for summary, formatted_code in zip(code_summaries, code_examples)]
                        
                        if settings.enable_batch_embeddings and len(embedding_texts) > 1:
                            logger.info(f"Creating embeddings for {len(embedding_texts)} code examples from {result_url} in single batch")
                            # Process all embeddings in a single batch call for better parallelization
                            code_embeddings = await embedding_service.create_embeddings_batch(embedding_texts)
                        else:
                            # Process embeddings individually
                            for embedding_text in embedding_texts:
                                embedding = await embedding_service.create_embedding(embedding_text)
                                code_embeddings.append(embedding)
                        
                        # Generate metadata
                        for i, code_block in enumerate(code_blocks):
                            code_metadatas.append({
                                'language': code_block['language'],
                                'char_count': len(code_block['code']),
                                'summary': code_summaries[i]
                            })
                        
                        # Store code examples
                        code_result = await database_service.add_code_examples(
                            urls=code_urls,
                            chunk_numbers=code_chunk_numbers,
                            code_examples=code_examples,
                            summaries=code_summaries,
                            embeddings=code_embeddings,
                            metadatas=code_metadatas
                        )
                        
                        code_examples_count = code_result.get('count', 0)
                
                return {
                    'url': result_url,
                    'chunks_created': chunks_created,
                    'code_examples': code_examples_count,
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"Error processing {result.get('url', 'unknown URL')}: {e}")
                return {
                    'url': result.get('url', 'unknown URL'),
                    'chunks_created': 0,
                    'code_examples': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Process all results in parallel for maximum efficiency
        logger.info(f"Processing {len(all_results)} URLs in parallel with batch operations")
        processing_tasks = [process_single_result(result) for result in all_results]
        processing_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Aggregate results
        total_chunks_created = 0
        total_code_examples = 0
        processed_urls = []
        failed_urls = []
        
        for result in processing_results:
            if isinstance(result, Exception):
                logger.error(f"Processing task failed with exception: {result}")
                failed_urls.append("unknown URL")
            elif result['success']:
                total_chunks_created += result['chunks_created']
                total_code_examples += result['code_examples']
                processed_urls.append(result['url'])
            else:
                failed_urls.append(result['url'])
                logger.error(f"Failed to process {result['url']}: {result.get('error', 'Unknown error')}")
        
        success_message = f"Successfully crawled {len(processed_urls)} URLs using {crawl_type} strategy"
        if failed_urls:
            success_message += f" ({len(failed_urls)} URLs failed)"
        
        return json.dumps({
            "success": True,
            "crawl_id": crawl_id,
            "crawl_type": crawl_type,
            "urls_processed": len(processed_urls),
            "urls_failed": len(failed_urls),
            "total_chunks_created": total_chunks_created,
            "total_code_examples": total_code_examples,
            "sources_updated": list(source_ids),
            "message": success_message
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "url": url
        })