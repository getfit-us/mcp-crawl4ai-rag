"""Tool for crawling a single page and storing in the database."""

import json
import logging
import asyncio
from urllib.parse import urlparse

from mcp.server.fastmcp import Context

from crawl4ai_mcp.mcp_server import mcp
from crawl4ai_mcp.models import CrawlContext
from crawl4ai_mcp.services.crawling import CrawlingService, CrawlCancelledException
from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.utilities.metadata import create_chunk_metadata
from crawl4ai_mcp.utilities.text_processing import TextProcessor

logger = logging.getLogger(__name__)


@mcp.tool()
async def crawl_single_page(
    ctx: Context,
    url: str
) -> str:
    """
    Crawl a single web page and store its content in PostgreSQL.
    
    This tool crawls a single web page, extracts its text content,
    chunks it, and stores it in PostgreSQL for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: The URL of the web page to crawl
        
    Returns:
        JSON response with summary of the crawling operation and storage in PostgreSQL
    """
    try:
        # Get context and services
        context: CrawlContext = ctx.request_context.lifespan_context
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
        
        # Parse domain for source identification
        source_id = urlparse(url).netloc
        
        logger.info(f"Starting crawl of single page: {url}")
        
        crawl_id = None
        
        try:
            # Start tracking the crawl operation
            crawl_id = await crawling_service.start_crawl_operation("single_page", [url])
            
            # Crawl the single page using batch method
            all_results = await crawling_service.crawl_batch([url], max_concurrent=1, crawl_id=crawl_id)
            
            if not all_results:
                return json.dumps({
                    "success": False,
                    "error": "Failed to crawl the URL",
                    "crawl_id": crawl_id,
                    "url": url
                })
            
            # Process the result
            result = all_results[0]
            result_url = result['url']
            markdown_content = result['markdown']
            
            # Chunk the content
            chunks = text_processor.smart_chunk_markdown(
                markdown_content, 
                chunk_size=settings.default_chunk_size
            )
            
            if not chunks:
                return json.dumps({
                    "success": False,
                    "error": "No content chunks were generated",
                    "crawl_id": crawl_id,
                    "url": url
                })
            
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
                    crawl_type="single_page",
                    section_info=section_info
                )
                metadatas.append(metadata)
            
            # Generate embeddings (batch or individual)
            if settings.use_contextual_embeddings:
                if settings.enable_batch_contextual_embeddings and len(chunks) > 1:
                    logger.info(f"Using batch contextual embeddings for {len(chunks)} chunks")
                    # Prepare chunks with documents for batch processing
                    chunks_with_docs = [(chunk, markdown_content) for chunk in chunks]
                    
                    # Process in batches - PARALLEL INSTEAD OF SEQUENTIAL
                    batch_size = settings.contextual_embedding_batch_size
                    contextual_tasks = []
                    
                    for i in range(0, len(chunks_with_docs), batch_size):
                        batch = chunks_with_docs[i:i + batch_size]
                        task = embedding_service.generate_contextual_embeddings_batch(batch)
                        contextual_tasks.append(task)
                    
                    # Execute all contextual embedding batches in parallel
                    contextual_batch_results = await asyncio.gather(*contextual_tasks)
                    contextual_results = []
                    for batch_result in contextual_batch_results:
                        contextual_results.extend(batch_result)
                    
                    # Get embeddings for contextual content
                    contextual_texts = [result[0] for result in contextual_results]
                    if settings.enable_batch_embeddings:
                        # Process embeddings in batches - PARALLEL INSTEAD OF SEQUENTIAL
                        embedding_batch_size = settings.embedding_batch_size
                        embedding_tasks = []
                        
                        for i in range(0, len(contextual_texts), embedding_batch_size):
                            batch_texts = contextual_texts[i:i + embedding_batch_size]
                            task = embedding_service.create_embeddings_batch(batch_texts)
                            embedding_tasks.append(task)
                        
                        # Execute all embedding batches in parallel
                        embedding_batch_results = await asyncio.gather(*embedding_tasks)
                        for batch_embeddings in embedding_batch_results:
                            embeddings.extend(batch_embeddings)
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
                    logger.info(f"Using batch embeddings for {len(chunks)} chunks")
                    # Process embeddings in batches - PARALLEL INSTEAD OF SEQUENTIAL
                    batch_size = settings.embedding_batch_size
                    embedding_tasks = []
                    
                    for i in range(0, len(chunks), batch_size):
                        batch_chunks = chunks[i:i + batch_size]
                        task = embedding_service.create_embeddings_batch(batch_chunks)
                        embedding_tasks.append(task)
                    
                    # Execute all embedding batches in parallel
                    embedding_batch_results = await asyncio.gather(*embedding_tasks)
                    for batch_embeddings in embedding_batch_results:
                        embeddings.extend(batch_embeddings)
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
            
            # Update source info
            total_word_count = len(markdown_content.split())
            source_summary = await crawling_service.extract_source_summary(
                source_id, markdown_content[:10000]
            )
            
            await database_service.update_source_info(
                source_id=source_id,
                summary=source_summary,
                word_count=total_word_count
            )
            
            # Extract and store code examples if enabled
            code_examples_created = 0
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
                        logger.info(f"Using batch summaries for {len(code_blocks)} code examples")
                        # Prepare data for batch processing
                        code_examples_data = [
                            (code_block['code'], code_block['context_before'], code_block['context_after'])
                            for code_block in code_blocks
                        ]
                        
                        # Process summaries in batches - PARALLEL INSTEAD OF SEQUENTIAL
                        batch_size = settings.summary_batch_size
                        summary_tasks = []
                        
                        for i in range(0, len(code_examples_data), batch_size):
                            batch_data = code_examples_data[i:i + batch_size]
                            task = crawling_service.generate_code_example_summaries_batch(batch_data)
                            summary_tasks.append(task)
                        
                        # Execute all summary batches in parallel
                        summary_batch_results = await asyncio.gather(*summary_tasks)
                        for batch_summaries in summary_batch_results:
                            code_summaries.extend(batch_summaries)
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
                        logger.info(f"Using batch embeddings for {len(embedding_texts)} code examples")
                        # Process embeddings in batches - PARALLEL INSTEAD OF SEQUENTIAL
                        batch_size = settings.embedding_batch_size
                        embedding_tasks = []
                        
                        for i in range(0, len(embedding_texts), batch_size):
                            batch_texts = embedding_texts[i:i + batch_size]
                            task = embedding_service.create_embeddings_batch(batch_texts)
                            embedding_tasks.append(task)
                        
                        # Execute all embedding batches in parallel
                        embedding_batch_results = await asyncio.gather(*embedding_tasks)
                        for batch_embeddings in embedding_batch_results:
                            code_embeddings.extend(batch_embeddings)
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
                    
                    code_examples_created = code_result.get('count', 0)
            
            logger.info(f"Successfully crawled and stored page: {url}")
            
            return json.dumps({
                "success": True,
                "crawl_id": crawl_id,
                "url": url,
                "source_id": source_id,
                "chunks_created": chunks_created,
                "code_examples_created": code_examples_created,
                "crawl_type": "single_page",
                "message": f"Successfully crawled and stored single page with {chunks_created} chunks"
            })
        
        except CrawlCancelledException as cancel_error:
            # Handle crawl cancellation specifically
            if crawl_id:
                await crawling_service.finish_crawl_operation(crawl_id, "cancelled")
            
            return json.dumps({
                "success": False,
                "error": "Crawl operation was cancelled",
                "crawl_id": crawl_id,
                "url": url,
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
        
    except Exception as e:
        logger.error(f"Error crawling single page {url}: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "url": url
        })