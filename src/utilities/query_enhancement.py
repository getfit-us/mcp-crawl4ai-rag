"""Query enhancement utilities for improved retrieval."""

import asyncio
import logging
import re
from typing import Any, List, Optional, Tuple
from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Utility class for query preprocessing and enhancement."""
    
    def __init__(self, settings: Optional[Any] = None, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize query enhancer.
        
        Args:
            settings: Application settings (optional)
            embedding_service: Embedding service instance (optional)
        """
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize query text.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        if not query or not query.strip():
            return ""
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters that don't add meaning
        query = re.sub(r'[^\w\s\-\.\?\!]', ' ', query)
        
        # Normalize case for common technical terms
        # Keep acronyms and technical terms in uppercase
        query = self._preserve_technical_terms(query)
        
        return query.strip()
    
    def _preserve_technical_terms(self, text: str) -> str:
        """
        Preserve common technical terms and acronyms in their standard case.
        
        Args:
            text: Input text
            
        Returns:
            Text with preserved technical terms
        """
        # Common technical terms to preserve
        tech_terms = {
            'api': 'API',
            'rest': 'REST', 
            'json': 'JSON',
            'xml': 'XML',
            'http': 'HTTP',
            'https': 'HTTPS',
            'sql': 'SQL',
            'css': 'CSS',
            'html': 'HTML',
            'javascript': 'JavaScript',
            'python': 'Python',
            'react': 'React',
            'vue': 'Vue',
            'angular': 'Angular',
            'node': 'Node',
            'npm': 'npm',
            'git': 'Git',
            'github': 'GitHub',
            'docker': 'Docker',
            'kubernetes': 'Kubernetes',
            'aws': 'AWS',
            'gcp': 'GCP',
            'azure': 'Azure'
        }
        
        words = text.split()
        processed_words = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in tech_terms:
                # Preserve the original punctuation
                if word != word_lower:
                    punctuation = word[len(word_lower):]
                    processed_words.append(tech_terms[word_lower] + punctuation)
                else:
                    processed_words.append(tech_terms[word_lower])
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to optimize retrieval strategy.
        
        Args:
            query: Query string
            
        Returns:
            Query type: 'factual', 'conceptual', 'code', 'how_to', 'troubleshooting'
        """
        query_lower = query.lower()
        
        # Code-related queries
        code_indicators = [
            'function', 'method', 'class', 'variable', 'import', 'export',
            'const', 'let', 'var', 'def', 'async', 'await', 'return',
            'if', 'else', 'for', 'while', 'loop', 'array', 'object',
            'string', 'number', 'boolean', 'null', 'undefined',
            'syntax', 'example code', 'code example', 'snippet'
        ]
        
        # How-to queries
        how_to_indicators = [
            'how to', 'how do i', 'how can i', 'tutorial', 'guide',
            'step by step', 'walkthrough', 'instructions', 'setup',
            'install', 'configure', 'create', 'build', 'implement'
        ]
        
        # Troubleshooting queries
        troubleshooting_indicators = [
            'error', 'issue', 'problem', 'bug', 'fix', 'solve',
            'troubleshoot', 'debug', 'not working', 'broken',
            'failed', 'exception', 'crash', 'stuck'
        ]
        
        # Conceptual queries
        conceptual_indicators = [
            'what is', 'what are', 'explain', 'concept', 'theory',
            'principle', 'difference between', 'comparison',
            'overview', 'introduction', 'basics', 'fundamentals'
        ]
        
        # Check for each type
        if any(indicator in query_lower for indicator in code_indicators):
            return 'code'
        elif any(indicator in query_lower for indicator in how_to_indicators):
            return 'how_to'
        elif any(indicator in query_lower for indicator in troubleshooting_indicators):
            return 'troubleshooting'
        elif any(indicator in query_lower for indicator in conceptual_indicators):
            return 'conceptual'
        else:
            return 'factual'
    
    def expand_query(self, query: str, query_type: str) -> List[str]:
        """
        Generate query variations for better retrieval coverage.
        
        Args:
            query: Original query
            query_type: Type of query detected
            
        Returns:
            List of query variations
        """
        variations = [query]  # Always include original
        
        if query_type == 'code':
            # Add code-specific variations
            variations.extend([
                f"code example {query}",
                f"implementation of {query}",
                f"syntax for {query}",
                f"{query} tutorial"
            ])
        
        elif query_type == 'how_to':
            # Add instructional variations
            variations.extend([
                query.replace('how to', 'tutorial for'),
                query.replace('how to', 'guide for'),
                query.replace('how do i', 'steps to'),
                f"example of {query}"
            ])
        
        elif query_type == 'troubleshooting':
            # Add problem-solving variations
            variations.extend([
                query.replace('error', 'issue'),
                query.replace('problem', 'error'),
                f"solution for {query}",
                f"fix {query}",
                f"resolve {query}"
            ])
        
        elif query_type == 'conceptual':
            # Add explanation variations
            variations.extend([
                query.replace('what is', 'explanation of'),
                query.replace('explain', 'overview of'),
                f"definition of {query}",
                f"introduction to {query}"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for variation in variations:
            if variation not in seen:
                seen.add(variation)
                unique_variations.append(variation)
        
        return unique_variations[:5]  # Limit to top 5 variations
    
    async def enhance_query_with_context(self, query: str, context: Optional[str] = None) -> str:
        """
        Enhance query using available context for better retrieval.
        
        Args:
            query: Original query
            context: Optional context information
            
        Returns:
            Enhanced query string
        """
        if not context:
            return query
        
        try:
            # Create a prompt to enhance the query with context
            prompt = f"""Given this context: {context[:500]}...

Original query: {query}

Enhance this query to be more specific and relevant for document retrieval. 
Return only the enhanced query, nothing else."""

            # Use the embedding service's client for enhancement
            response = await self.embedding_service._run_in_executor(
                lambda: self.embedding_service.client.chat.completions.create(
                    model=self.settings.summary_llm_model,
                    messages=[
                        {"role": "system", "content": "You are a query enhancement assistant. Improve queries for better document retrieval."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100
                )
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            
            # Validate the enhanced query
            if enhanced_query and len(enhanced_query) > 0 and len(enhanced_query.split()) >= len(query.split()):
                return enhanced_query
            else:
                return query
        
        except Exception as e:
            logger.error(f"Error enhancing query with context: {e}")
            return query
    
    async def generate_multi_queries(self, query: str, count: int = 3) -> List[str]:
        """
        Generate multiple variations of a query for comprehensive retrieval.
        
        Args:
            query: Original query
            count: Number of query variations to generate
            
        Returns:
            List of query variations including the original
        """
        try:
            prompt = f"""Generate {count-1} alternative phrasings of this query for document retrieval:

Original query: {query}

Return only the alternative queries, one per line, without numbering or explanation."""

            response = await self.embedding_service._run_in_executor(
                lambda: self.embedding_service.client.chat.completions.create(
                    model=self.settings.summary_llm_model,
                    messages=[
                        {"role": "system", "content": "You are a query reformulation assistant. Generate alternative phrasings for better document retrieval."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
            )
            
            alternative_queries = response.choices[0].message.content.strip().split('\n')
            alternative_queries = [q.strip() for q in alternative_queries if q.strip()]
            
            # Combine with original and remove duplicates
            all_queries = [query] + alternative_queries
            seen = set()
            unique_queries = []
            for q in all_queries:
                if q not in seen and len(q.split()) >= 2:  # Ensure meaningful queries
                    seen.add(q)
                    unique_queries.append(q)
            
            return unique_queries[:count]
        
        except Exception as e:
            logger.error(f"Error generating multi-queries: {e}")
            return [query]
    
    async def process_query(
        self, 
        query: str, 
        context: Optional[str] = None,
        enable_multi_query: bool = False,
        enable_expansion: bool = True
    ) -> Tuple[str, List[str], str]:
        """
        Comprehensive query processing pipeline.
        
        Args:
            query: Original query
            context: Optional context for enhancement
            enable_multi_query: Whether to generate multiple query variations
            enable_expansion: Whether to expand query based on type
            
        Returns:
            Tuple of (enhanced_main_query, additional_queries, query_type)
        """
        # Step 1: Clean the query
        cleaned_query = self.clean_query(query)
        if not cleaned_query:
            return query, [], 'factual'
        
        # Step 2: Detect query type
        query_type = self.detect_query_type(cleaned_query)
        
        # Step 3: Enhance with context if available
        if context:
            enhanced_query = await self.enhance_query_with_context(cleaned_query, context)
        else:
            enhanced_query = cleaned_query
        
        # Step 4: Generate additional queries
        additional_queries = []
        
        if enable_expansion:
            # Add type-based expansions
            expansions = self.expand_query(enhanced_query, query_type)
            additional_queries.extend(expansions[1:])  # Skip the original
        
        if enable_multi_query:
            # Add AI-generated variations
            ai_variations = await self.generate_multi_queries(enhanced_query, 3)
            additional_queries.extend(ai_variations[1:])  # Skip the original
        
        # Remove duplicates
        seen = {enhanced_query}
        unique_additional = []
        for q in additional_queries:
            if q not in seen:
                seen.add(q)
                unique_additional.append(q)
        
        return enhanced_query, unique_additional[:5], query_type  # Limit additional queries