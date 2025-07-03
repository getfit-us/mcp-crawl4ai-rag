"""Maximum Marginal Relevance (MMR) algorithm for result diversification."""

import logging
import numpy as np
from typing import List, Optional, Tuple, Union
from crawl4ai_mcp.models import SearchResult
from crawl4ai_mcp.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class MMRDiversifier:
    """Maximum Marginal Relevance algorithm for search result diversification."""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize MMR diversifier.
        
        Args:
            embedding_service: Service for generating embeddings
        """
        self.embedding_service = embedding_service
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec_a = np.array(a)
            vec_b = np.array(b)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def calculate_content_similarity(self, content_a: str, content_b: str, threshold: float = 0.7) -> float:
        """
        Calculate similarity between two content strings using simple text overlap.
        
        Args:
            content_a: First content string
            content_b: Second content string
            threshold: Threshold for considering content similar
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize and tokenize
            words_a = set(content_a.lower().split())
            words_b = set(content_b.lower().split())
            
            if not words_a or not words_b:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(words_a.intersection(words_b))
            union = len(words_a.union(words_b))
            
            return intersection / union if union > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    async def get_embeddings_for_results(self, results: List[SearchResult]) -> List[Optional[List[float]]]:
        """
        Get or generate embeddings for search results.
        
        Args:
            results: List of search results
            
        Returns:
            List of embeddings (None for failed generations)
        """
        embeddings = []
        
        # Extract content for batch embedding generation
        contents = [result.content[:1000] for result in results]  # Limit content length
        
        try:
            # Generate embeddings in batch for efficiency
            batch_embeddings = await self.embedding_service.create_embeddings_batch(contents)
            return batch_embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings for MMR: {e}")
            # Fallback to individual embeddings
            for content in contents:
                try:
                    embedding = await self.embedding_service.create_embedding(content)
                    embeddings.append(embedding)
                except Exception as ind_e:
                    logger.error(f"Error generating individual embedding: {ind_e}")
                    embeddings.append(None)
            
            return embeddings
    
    async def mmr_rerank(
        self,
        query: str,
        results: List[SearchResult],
        lambda_param: float = 0.7,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Apply Maximum Marginal Relevance algorithm to diversify search results.
        
        Args:
            query: Original search query
            results: List of search results to diversify
            lambda_param: Balance between relevance and diversity (0.0 = max diversity, 1.0 = max relevance)
            top_k: Number of results to return (default: same as input)
            
        Returns:
            Reranked and diversified list of search results
        """
        if not results:
            return results
        
        if top_k is None:
            top_k = len(results)
        
        top_k = min(top_k, len(results))
        
        try:
            # Step 1: Generate query embedding
            query_embedding = await self.embedding_service.create_embedding(query)
            
            # Step 2: Generate embeddings for all results
            result_embeddings = await self.get_embeddings_for_results(results)
            
            # Step 3: Calculate initial relevance scores
            relevance_scores = []
            for i, embedding in enumerate(result_embeddings):
                if embedding is not None:
                    # Use embedding similarity
                    relevance = self.cosine_similarity(query_embedding, embedding)
                else:
                    # Fallback to existing similarity score
                    relevance = results[i].similarity_score
                
                relevance_scores.append(relevance)
            
            # Step 4: MMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(results)))
            
            # Select the first result (highest relevance)
            if remaining_indices:
                best_idx = max(remaining_indices, key=lambda i: relevance_scores[i])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            
            # Iteratively select remaining results
            while len(selected_indices) < top_k and remaining_indices:
                best_score = float('-inf')
                best_idx = None
                
                for candidate_idx in remaining_indices:
                    # Calculate relevance score
                    relevance = relevance_scores[candidate_idx]
                    
                    # Calculate maximum similarity to already selected results
                    max_similarity = 0.0
                    
                    for selected_idx in selected_indices:
                        # Use embedding similarity if available
                        if (result_embeddings[candidate_idx] is not None and 
                            result_embeddings[selected_idx] is not None):
                            similarity = self.cosine_similarity(
                                result_embeddings[candidate_idx],
                                result_embeddings[selected_idx]
                            )
                        else:
                            # Fallback to content similarity
                            similarity = self.calculate_content_similarity(
                                results[candidate_idx].content,
                                results[selected_idx].content
                            )
                        
                        max_similarity = max(max_similarity, similarity)
                    
                    # Calculate MMR score
                    mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = candidate_idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
            
            # Step 5: Return diversified results
            diversified_results = [results[i] for i in selected_indices]
            
            # Add MMR scores to results for debugging
            for i, result in enumerate(diversified_results):
                result.mmr_rank = i + 1
            
            logger.info(f"MMR diversification: {len(results)} -> {len(diversified_results)} results")
            return diversified_results
        
        except Exception as e:
            logger.error(f"Error in MMR reranking: {e}")
            # Return original results if MMR fails
            return results[:top_k] if top_k else results
    
    async def diversify_by_content_type(
        self,
        results: List[SearchResult],
        max_per_type: int = 2
    ) -> List[SearchResult]:
        """
        Diversify results by content type to ensure variety.
        
        Args:
            results: List of search results
            max_per_type: Maximum results per content type
            
        Returns:
            Diversified list ensuring variety in content types
        """
        if not results:
            return results
        
        # Group results by content type (detected from metadata or content)
        type_groups = {}
        
        for result in results:
            content_type = self._detect_content_type(result)
            if content_type not in type_groups:
                type_groups[content_type] = []
            type_groups[content_type].append(result)
        
        # Select top results from each type
        diversified_results = []
        
        # Prioritize types by their best scores
        type_scores = {}
        for content_type, group in type_groups.items():
            type_scores[content_type] = max(r.similarity_score for r in group)
        
        sorted_types = sorted(type_scores.keys(), key=lambda t: type_scores[t], reverse=True)
        
        # Round-robin selection from each type
        type_indices = {t: 0 for t in sorted_types}
        
        while len(diversified_results) < len(results):
            added_any = False
            
            for content_type in sorted_types:
                group = type_groups[content_type]
                idx = type_indices[content_type]
                
                if idx < len(group) and idx < max_per_type:
                    diversified_results.append(group[idx])
                    type_indices[content_type] += 1
                    added_any = True
            
            if not added_any:
                break
        
        logger.info(f"Content type diversification: {len(type_groups)} types identified")
        return diversified_results
    
    def _detect_content_type(self, result: SearchResult) -> str:
        """
        Detect content type from result metadata or content.
        
        Args:
            result: Search result to analyze
            
        Returns:
            Content type string
        """
        # Check metadata first
        if 'content_type' in result.metadata:
            return result.metadata['content_type']
        
        # Analyze content for type detection
        content_lower = result.content.lower()
        
        # Code patterns
        code_patterns = [
            'function', 'def ', 'class ', 'import ', 'const ', 'let ', 'var ',
            '```', 'code example', 'snippet', 'syntax'
        ]
        
        # Tutorial patterns
        tutorial_patterns = [
            'step 1', 'step 2', 'tutorial', 'guide', 'walkthrough',
            'how to', 'instructions', 'follow these steps'
        ]
        
        # API patterns
        api_patterns = [
            'api', 'endpoint', 'request', 'response', 'http',
            'get ', 'post ', 'put ', 'delete ', 'rest'
        ]
        
        # Configuration patterns
        config_patterns = [
            'config', 'configuration', 'settings', 'options',
            'setup', 'install', 'environment'
        ]
        
        if any(pattern in content_lower for pattern in code_patterns):
            return 'code'
        elif any(pattern in content_lower for pattern in tutorial_patterns):
            return 'tutorial'
        elif any(pattern in content_lower for pattern in api_patterns):
            return 'api'
        elif any(pattern in content_lower for pattern in config_patterns):
            return 'configuration'
        else:
            return 'general'
    
    async def apply_diversification(
        self,
        query: str,
        results: List[SearchResult],
        diversification_strategy: str = 'mmr',
        lambda_param: float = 0.7,
        max_per_type: int = 3,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Apply diversification strategy to search results.
        
        Args:
            query: Original search query
            results: List of search results
            diversification_strategy: 'mmr', 'content_type', or 'hybrid'
            lambda_param: MMR parameter for relevance vs diversity balance
            max_per_type: Maximum results per content type
            top_k: Number of results to return
            
        Returns:
            Diversified search results
        """
        if not results:
            return results
        
        try:
            if diversification_strategy == 'mmr':
                return await self.mmr_rerank(query, results, lambda_param, top_k)
            
            elif diversification_strategy == 'content_type':
                diversified = await self.diversify_by_content_type(results, max_per_type)
                return diversified[:top_k] if top_k else diversified
            
            elif diversification_strategy == 'hybrid':
                # First apply content type diversification, then MMR
                type_diversified = await self.diversify_by_content_type(results, max_per_type)
                return await self.mmr_rerank(query, type_diversified, lambda_param, top_k)
            
            else:
                logger.warning(f"Unknown diversification strategy: {diversification_strategy}")
                return results[:top_k] if top_k else results
        
        except Exception as e:
            logger.error(f"Error applying diversification: {e}")
            return results[:top_k] if top_k else results