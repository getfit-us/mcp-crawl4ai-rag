"""Configuration management using Pydantic settings."""

from typing import Optional
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation."""

    # MCP Server Configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8051"))
    transport: str = os.getenv(
        "TRANSPORT", "streamable-http"
    )  # Options: "stdio", "sse", "streamable-http"

    # LLM Configuration - REQUIRED
    openai_api_key: str = os.getenv("LLM_MODEL_API_KEY", "test-key")
    openai_base_url: Optional[str] = os.getenv("SUMMARY_LLM_BASE_URL")  # Custom base URL for OpenAI-compatible endpoints
    openai_organization: Optional[str] = os.getenv("OPENAI_ORGANIZATION")  # Organization ID for OpenAI
    summary_llm_model: str = os.getenv("SUMMARY_LLM_MODEL", "gpt-4o-mini")
    disable_thinking: bool = os.getenv("DISABLE_THINKING", "false") == "true"  # Disable thinking on think models (adds /no_think to prompt)

    # PostgreSQL Configuration - REQUIRED if using Local Postgres instead of Supabase
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "crawl4ai_rag")
    postgres_user: str = os.getenv("POSTGRES_USER", "")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_sslmode: str = os.getenv("POSTGRES_SSLMODE", "prefer")

    # Feature Flags
    use_contextual_embeddings: bool = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    use_hybrid_search: bool = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
    use_reranking: bool = os.getenv("USE_RERANKING", "false") == "true"
    use_agentic_rag: bool = os.getenv("USE_AGENTIC_RAG", "false") == "true"  # For code example extraction

    # Crawling Configuration
    default_max_depth: int = 3
    default_max_concurrent: int = 10
    default_chunk_size: int = 5000
    default_overlap: int = 200

    # Browser Cleanup Configuration (focused only on process cleanup, not operation timeouts)
    browser_headless: bool = os.getenv("BROWSER_HEADLESS", "true") == "true"
    browser_cleanup_timeout: int = int(os.getenv("BROWSER_CLEANUP_TIMEOUT", "10"))  # seconds for shutdown cleanup
    browser_process_isolation: bool = os.getenv("BROWSER_PROCESS_ISOLATION", "false") == "true"  # Enable for easier cleanup
    browser_auto_cleanup: bool = os.getenv("BROWSER_AUTO_CLEANUP", "true") == "true"  # Enable background cleanup

    # Search Configuration
    default_num_results: int = 5
    default_semantic_threshold: float = 0.5
    default_rerank_threshold: float = 0.3

    # Embedding Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    embedding_service_type: str = os.getenv("EMBEDDING_SERVICE_TYPE", "huggingface")  # "openai", "huggingface", "custom"
    custom_embedding_url: Optional[str] = os.getenv("CUSTOM_EMBEDDING_URL")  
    embedding_api_key: Optional[str] = os.getenv("EMBEDDING_API_KEY")
    embedding_organization: Optional[str] = os.getenv("EMBEDDING_ORGANIZATION")
    
    # Token limits for embedding models
    embedding_max_tokens: int = int(os.getenv("EMBEDDING_MAX_TOKENS", "8000"))  # Safe limit below 8196
    embedding_chars_per_token: float = float(os.getenv("EMBEDDING_CHARS_PER_TOKEN", "4.0"))  # Approx chars per token

    # Batch Processing Configuration
    enable_batch_embeddings: bool = os.getenv("ENABLE_BATCH_EMBEDDINGS", "true") == "true"
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    enable_batch_summaries: bool = os.getenv("ENABLE_BATCH_SUMMARIES", "true") == "true"
    summary_batch_size: int = int(os.getenv("SUMMARY_BATCH_SIZE", "10"))
    enable_batch_contextual_embeddings: bool = os.getenv("ENABLE_BATCH_CONTEXTUAL_EMBEDDINGS", "true") == "true"
    contextual_embedding_batch_size: int = int(os.getenv("CONTEXTUAL_EMBEDDING_BATCH_SIZE", "20"))

    # Reranking Configuration
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    custom_reranker_url: Optional[str] = os.getenv("CUSTOM_RERANKER_URL", None)  # For custom reranking model URLs
    reranker_model_local_path: Optional[str] = os.getenv("RERANKER_MODEL_LOCAL_PATH", None)  

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def validate_required_fields(self) -> None:
        """Validate that all required fields are set."""
        if not self.openai_api_key:
            raise ValueError("LLM_MODEL_API_KEY is required")
        if not self.postgres_user:
            raise ValueError("POSTGRES_USER is required")
        if not self.postgres_password:
            raise ValueError("POSTGRES_PASSWORD is required")

        # Validate transport type
        valid_transports = {"stdio", "sse", "streamable-http"}
        if self.transport not in valid_transports:
            raise ValueError(f"TRANSPORT must be one of {valid_transports}, got: {self.transport}")

    @property
    def postgres_dsn(self) -> str:
        """Get PostgreSQL connection string."""
        # URL-encode the username and password to handle special characters
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql://{user}:{password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode}"
        )


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global settings
    if settings is None:
        settings = Settings()  # type: ignore[call-arg]  # BaseSettings loads from env
        settings.validate_required_fields()
    return settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global settings
    settings = None
