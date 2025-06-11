"""Reranking utilities for search result optimization."""

import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from crawl4ai_mcp.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Utility class for reranking search results."""
    
    def __init__(
        self, 
        model: Optional[CrossEncoder] = None, 
        settings: Optional[Any] = None
    ):
        """
        Initialize reranker.
        
        Args:
            model: CrossEncoder model instance (optional)
            settings: Application settings (optional)
        """
        self.settings = settings or get_settings()
        self.model = model
        
        # Initialize model if not provided and reranking is enabled
        if self.model is None and self.settings.use_reranking:
            model_path = self._get_model_path()
            if model_path:
                try:
                    self.model = CrossEncoder(model_path)
                    logger.info(f"Loaded reranking model from: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load reranking model from {model_path}: {e}")
                    logger.info(f"Falling back to default model: {self.settings.cross_encoder_model}")
                    self.model = CrossEncoder(self.settings.cross_encoder_model)
            else:
                self.model = CrossEncoder(self.settings.cross_encoder_model)
    
    def _get_model_path(self) -> Optional[str]:
        """
        Determine the model path based on configuration settings.
        
        Returns:
            Model path (URL, local path, or None to use default)
        """
        # Check for local path first (highest priority)
        if hasattr(self.settings, 'cross_encoder_model_local_path') and self.settings.cross_encoder_model_local_path:
            return self.settings.cross_encoder_model_local_path
        
        # Check for custom URL (second priority)
        if hasattr(self.settings, 'custom_cross_encoder_url') and self.settings.custom_cross_encoder_url:
            return self.settings.custom_cross_encoder_url
        
        # Return None to use default model
        return None
    
    def rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        content_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder model.
        
        Args:
            query: The search query
            results: List of search results
            content_key: The key in each result dict that contains the text content
            
        Returns:
            Reranked list of results
        """
        if not self.model or not results:
            return results
        
        try:
            # Extract content from results
            texts = [result.get(content_key, "") for result in results]
            
            # Create pairs of [query, document] for the cross-encoder
            pairs = [[query, text] for text in texts]
            
            # Get relevance scores from the cross-encoder
            scores = self.model.predict(pairs)
            
            # Add scores to results and sort by score (descending)
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return results
    
    def filter_by_threshold(
        self,
        results: List[Dict[str, Any]],
        threshold: float,
        score_key: str = "rerank_score"
    ) -> List[Dict[str, Any]]:
        """
        Filter results by minimum score threshold.
        
        Args:
            results: List of results with scores
            threshold: Minimum score threshold
            score_key: The key containing the score
            
        Returns:
            Filtered list of results
        """
        return [
            result for result in results 
            if result.get(score_key, 0) >= threshold
        ]