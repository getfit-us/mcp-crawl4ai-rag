"""Database service for PostgreSQL operations with pgvector."""

import logging
from typing import Any, Dict, List, Optional

import asyncpg
from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import SourceInfo

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing database operations with PostgreSQL."""
    
    def __init__(self, pool: asyncpg.Pool, settings: Optional[Any] = None):
        """
        Initialize database service.
        
        Args:
            pool: AsyncPG connection pool
            settings: Application settings (optional)
        """
        self.pool = pool
        self.settings = get_settings()
    
    async def add_documents(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        url_to_full_document: Dict[str, str],
        batch_size: int = 20
    ) -> Dict[str, Any]:
        """
        Add documents to the PostgreSQL crawled_pages table in batches.
        
        Args:
            urls: List of URLs
            chunk_numbers: List of chunk numbers
            contents: List of document contents
            embeddings: List of embeddings for each content
            metadatas: List of document metadata
            url_to_full_document: Dictionary mapping URLs to their full document content
            batch_size: Size of each batch for insertion
            
        Returns:
            Result dictionary with success status and document count
        """
        # Get unique URLs to delete existing records
        unique_urls = list(set(urls))
        
        # Extract unique source domains
        from urllib.parse import urlparse
        unique_sources = list(set([urlparse(url).netloc for url in unique_urls]))
        
        async with self.pool.acquire() as conn:
            # Create source records if they don't exist
            try:
                for source_id in unique_sources:
                    await conn.execute(
                        """
                        INSERT INTO sources (source_id, summary, total_word_count)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (source_id) DO NOTHING
                        """,
                        source_id, f"Auto-created source for {source_id}", 0
                    )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create source records: {str(e)}",
                    "count": 0
                }
            
            # Delete existing records for these URLs
            try:
                if unique_urls:
                    await conn.execute(
                        "DELETE FROM crawled_pages WHERE url = ANY($1)",
                        unique_urls
                    )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to delete existing records: {str(e)}",
                    "count": 0
                }
            
            # Insert documents in batches
            total_documents = len(urls)
            documents_added = 0
            
            for i in range(0, total_documents, batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_chunk_numbers = chunk_numbers[i:i + batch_size]
                batch_contents = contents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                # Extract source domains
                batch_sources = [urlparse(url).netloc for url in batch_urls]
                
                # Prepare batch data for insertion
                batch_data = []
                for j, (url, chunk_num, content, embedding, metadata, source) in enumerate(zip(
                    batch_urls, batch_chunk_numbers, batch_contents, 
                    batch_embeddings, batch_metadatas, batch_sources
                )):
                    batch_data.append((
                        url, chunk_num, content, metadata, source, embedding
                    ))
                
                # Insert batch
                try:
                    if batch_data:
                        await conn.executemany(
                            """
                            INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            """,
                            batch_data
                        )
                        documents_added += len(batch_data)
                except Exception as e:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                    continue
        
        return {
            "success": True,
            "count": documents_added,
            "total": total_documents
        }
    
    async def add_code_examples(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        code_examples: List[str],
        summaries: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 20
    ) -> Dict[str, Any]:
        """
        Add code examples to the PostgreSQL code_examples table in batches.
        
        Args:
            urls: List of URLs
            chunk_numbers: List of chunk numbers
            code_examples: List of code example contents
            summaries: List of code example summaries
            embeddings: List of embeddings for each summary
            metadatas: List of metadata dictionaries
            batch_size: Size of each batch for insertion
            
        Returns:
            Result dictionary with success status and example count
        """
        if not urls:
            return {"success": True, "count": 0}
        
        # Extract unique source domains
        from urllib.parse import urlparse
        unique_urls = list(set(urls))
        unique_sources = list(set([urlparse(url).netloc for url in unique_urls]))
        
        async with self.pool.acquire() as conn:
            # Create source records if they don't exist
            try:
                for source_id in unique_sources:
                    await conn.execute(
                        """
                        INSERT INTO sources (source_id, summary, total_word_count)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (source_id) DO NOTHING
                        """,
                        source_id, f"Auto-created source for {source_id}", 0
                    )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create source records: {str(e)}",
                    "count": 0
                }
            
            # Delete existing records for these URLs
            for url in unique_urls:
                try:
                    await conn.execute("DELETE FROM code_examples WHERE url = $1", url)
                except Exception as e:
                    logger.error(f"Error deleting existing code examples for {url}: {e}")
            
            # Process code examples in batches
            total_examples = len(urls)
            examples_added = 0
            
            for i in range(0, total_examples, batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_chunk_numbers = chunk_numbers[i:i + batch_size]
                batch_code_examples = code_examples[i:i + batch_size]
                batch_summaries = summaries[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                # Extract source domains
                batch_sources = [urlparse(url).netloc for url in batch_urls]
                
                # Prepare batch data
                batch_data = []
                for url, chunk_num, code, summary, embedding, metadata, source in zip(
                    batch_urls, batch_chunk_numbers, batch_code_examples,
                    batch_summaries, batch_embeddings, batch_metadatas, batch_sources
                ):
                    # Extract language from code block if available
                    language = "unknown"
                    if code.startswith("```"):
                        first_line = code.split('\n')[0]
                        if len(first_line) > 3:
                            language = first_line[3:].strip()
                    
                    # Add language to metadata
                    metadata['language'] = language
                    
                    batch_data.append((
                        url, chunk_num, code, summary, metadata, source, embedding
                    ))
                
                # Insert batch
                try:
                    if batch_data:
                        await conn.executemany(
                            """
                            INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """,
                            batch_data
                        )
                        examples_added += len(batch_data)
                except Exception as e:
                    logger.error(f"Error inserting code examples batch {i//batch_size + 1}: {e}")
                    continue
        
        return {
            "success": True,
            "count": examples_added,
            "total": total_examples
        }
    
    async def update_source_info(
        self,
        source_id: str,
        summary: str,
        word_count: int
    ) -> Dict[str, Any]:
        """
        Update or insert source information in the sources table.
        
        Args:
            source_id: The source ID (domain)
            summary: Summary of the source
            word_count: Total word count for the source
            
        Returns:
            Result dictionary with success status
        """
        async with self.pool.acquire() as conn:
            try:
                # Try to update existing source
                result = await conn.execute(
                    """
                    UPDATE sources 
                    SET summary = $2, total_word_count = $3, updated_at = CURRENT_TIMESTAMP
                    WHERE source_id = $1
                    """,
                    source_id, summary, word_count
                )
                
                # If no rows were updated, insert new source
                if result == "UPDATE 0":
                    await conn.execute(
                        """
                        INSERT INTO sources (source_id, summary, total_word_count)
                        VALUES ($1, $2, $3)
                        """,
                        source_id, summary, word_count
                    )
                    logger.info(f"Created new source: {source_id}")
                else:
                    logger.info(f"Updated source: {source_id}")
                    
                return {"success": True, "source_id": source_id}
                
            except Exception as e:
                logger.error(f"Error updating source info: {e}")
                return {"success": False, "error": str(e)}
    
    async def check_source_exists(self, source_id: str) -> Dict[str, Any]:
        """
        Check if a source already has content in the database.
        
        Args:
            source_id: The source ID (domain) to check
            
        Returns:
            Dictionary with existence info and statistics
        """
        async with self.pool.acquire() as conn:
            try:
                # Check if source exists and get stats
                result = await conn.fetchrow("""
                    SELECT 
                        s.source_id,
                        s.summary,
                        s.total_word_count,
                        s.updated_at,
                        s.created_at,
                        COALESCE(cp_stats.doc_count, 0) as doc_count,
                        COALESCE(cp_stats.chunk_count, 0) as chunk_count,
                        COALESCE(ce_stats.code_count, 0) as code_count
                    FROM sources s
                    LEFT JOIN (
                        SELECT 
                            source_id,
                            COUNT(DISTINCT url) as doc_count,
                            COUNT(id) as chunk_count
                        FROM crawled_pages
                        GROUP BY source_id
                    ) cp_stats ON s.source_id = cp_stats.source_id
                    LEFT JOIN (
                        SELECT 
                            source_id,
                            COUNT(DISTINCT url) as code_count
                        FROM code_examples
                        GROUP BY source_id
                    ) ce_stats ON s.source_id = ce_stats.source_id
                    WHERE s.source_id = $1
                """, source_id)
                
                if result:
                    return {
                        "exists": True,
                        "source_id": result['source_id'],
                        "total_documents": result['doc_count'] or 0,
                        "total_chunks": result['chunk_count'] or 0,
                        "total_code_examples": result['code_count'] or 0,
                        "word_count": result['total_word_count'] or 0,
                        "last_updated": result['updated_at'] or result['created_at'],
                        "summary": result['summary']
                    }
                else:
                    return {"exists": False, "source_id": source_id}
                    
            except Exception as e:
                logger.error(f"Error checking source existence: {e}")
                return {"exists": False, "source_id": source_id, "error": str(e)}

    async def get_available_sources(self) -> List[SourceInfo]:
        """
        Get all available sources from the database.
        
        Returns:
            List of SourceInfo objects
        """
        async with self.pool.acquire() as conn:
            try:
                # Get sources with document counts (using subqueries to avoid Cartesian product issues)
                sources_query = """
                    SELECT 
                        s.source_id,
                        s.summary,
                        s.total_word_count,
                        s.updated_at,
                        s.created_at,
                        COALESCE(cp_stats.doc_count, 0) as doc_count,
                        COALESCE(cp_stats.chunk_count, 0) as chunk_count,
                        COALESCE(ce_stats.code_count, 0) as code_count
                    FROM sources s
                    LEFT JOIN (
                        SELECT 
                            source_id,
                            COUNT(DISTINCT url) as doc_count,
                            COUNT(id) as chunk_count
                        FROM crawled_pages
                        GROUP BY source_id
                    ) cp_stats ON s.source_id = cp_stats.source_id
                    LEFT JOIN (
                        SELECT 
                            source_id,
                            COUNT(DISTINCT url) as code_count
                        FROM code_examples
                        GROUP BY source_id
                    ) ce_stats ON s.source_id = ce_stats.source_id
                """
                
                rows = await conn.fetch(sources_query)
                sources = []
                
                for row in rows:
                    sources.append(SourceInfo(
                        source=row['source_id'],
                        total_documents=row['doc_count'] or 0,
                        total_chunks=row['chunk_count'] or 0,
                        total_code_examples=row['code_count'] or 0,
                        word_count=row['total_word_count'] or 0,
                        last_updated=row['updated_at'] or row['created_at'],
                        summary=row['summary']
                    ))
                
                return sources
                
            except Exception as e:
                logger.error(f"Error getting available sources: {e}")
                return []
    
    
    def _generate_contextual_content(
        self,
        chunk_content: str,
        full_document: str,
        chunk_number: int
    ) -> str:
        """
        Generate contextual content for a chunk.
        
        Args:
            chunk_content: The content of the current chunk
            full_document: The full document content
            chunk_number: The chunk number
            
        Returns:
            Contextual content string
        """
        # This is a simplified version - the actual implementation
        # would be in the embeddings service
        return f"Chunk {chunk_number} of document: {chunk_content}"