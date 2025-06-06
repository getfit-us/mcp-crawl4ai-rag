# Refactor Plan

## Tree Structure

```
mcp-crawl4ai-rag/
├── main.py                    # Entry point (minimal - just runs src.mcp_server)
├── pyproject.toml
├── Dockerfile
├── crawled_pages.sql
├── .env.example
├── README.md
├── QUICKSTART.md
├── ONBOARDING.md
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic settings & environment validation
│   ├── models.py               # All Pydantic models for requests/responses
│   ├── mcp_server.py           # FastMCP server setup & lifespan management
│   │
│   ├── services/               # Business logic layer
│   │   ├── __init__.py
│   │   ├── crawling.py         # All crawling logic
│   │   ├── search.py           # Search/RAG logic
│   │   ├── database.py         # Database operations
│   │   └── embeddings.py       # Embedding operations
│   │
│   ├── tools/                  # MCP tool definitions (thin wrappers)
│   │   ├── __init__.py
│   │   ├── crawl_single_page.py
│   │   ├── smart_crawl_url.py
│   │   ├── get_available_sources.py
│   │   ├── perform_rag_query.py
│   │   ├── search_code_examples.py
│   │   │
│   │   └── tests/              # Unit tests for each tool
│   │       ├── __init__.py
│   │       ├── test_crawl_single_page.py
│   │       ├── test_smart_crawl_url.py
│   │       ├── test_get_available_sources.py
│   │       ├── test_perform_rag_query.py
│   │       └── test_search_code_examples.py
│   │
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── text_processing.py  # Text chunking & extraction
│       └── reranking.py        # Cross-encoder reranking
│
└── tests/                      # Integration tests
    ├── __init__.py
    ├── conftest.py
    └── test_integration.py
```

## Detailed Mappings from Current Code

### From `src/crawl4ai_mcp.py`:

| Current Function/Class | New Location | Notes |
|------------------------|--------------|-------|
| `FastMCP` initialization | `src/mcp_server.py` | Server setup |
| `Crawl4AIContext` | `src/models.py` | As Pydantic model |
| `crawl4ai_lifespan()` | `src/mcp_server.py` | Lifespan management |
| `rerank_results()` | `src/utils/reranking.py` | Utility function |
| `is_sitemap()`, `is_txt()` | `src/services/crawling.py` | URL detection methods |
| `parse_sitemap()` | `src/services/crawling.py` | Sitemap parsing |
| `smart_chunk_markdown()` | `src/utils/text_processing.py` | Text chunking |
| `extract_section_info()` | `src/utils/text_processing.py` | Metadata extraction |
| `process_code_example()` | `src/services/crawling.py` | Code processing |
| `@mcp.tool` decorated functions | `src/tools/*.py` | Each in its own file |
| `crawl_batch()` | `src/services/crawling.py` | Batch crawling |
| `crawl_recursive_internal_links()` | `src/services/crawling.py` | Recursive crawling |
| `crawl_markdown_file()` | `src/services/crawling.py` | Text file crawling |

### From `src/utils.py`:

| Current Function | New Location | Notes |
|------------------|--------------|-------|
| `get_supabase_client()` | `src/services/database.py` | Database client factory |
| `create_embedding()` | `src/services/embeddings.py` | Single embedding |
| `create_embeddings_batch()` | `src/services/embeddings.py` | Batch embeddings |
| `generate_contextual_embedding()` | `src/services/embeddings.py` | Contextual embeddings |
| `process_chunk_with_context()` | `src/services/embeddings.py` | Context processing |
| `add_documents_to_supabase()` | `src/services/database.py` | Document storage |
| `search_documents()` | `src/services/search.py` | Vector search |
| `extract_code_blocks()` | `src/services/crawling.py` | Code extraction |
| `generate_code_example_summary()` | `src/services/crawling.py` | Code summarization |
| `add_code_examples_to_supabase()` | `src/services/database.py` | Code example storage |
| `update_source_info()` | `src/services/database.py` | Source updates |
| `extract_source_summary()` | `src/services/crawling.py` | Source summarization |
| `search_code_examples()` | `src/services/search.py` | Code example search |

## New Files to Create

### 1. `src/config.py`
```python
# Environment configuration with validation
# - All os.getenv() calls move here
# - Pydantic BaseSettings for validation
# - Single source of truth for config
```

### 2. `src/models.py`
```python
# All Pydantic models:
# - CrawlRequest, CrawlResult
# - SearchRequest, SearchResult
# - CodeExample, Source
# - Crawl4AIContext (from dataclass)
# - Any other data structures
```

### 3. `src/mcp_server.py`
```python
# MCP server setup:
# - FastMCP initialization
# - Lifespan context manager
# - Server startup logic
# - NO tool definitions (those go in tools/)
```

### 4. Service Files
Each service file contains related business logic:

- **`services/crawling.py`**: All crawling strategies, URL detection, content processing
- **`services/search.py`**: Vector search, hybrid search, search result processing
- **`services/database.py`**: All Supabase operations, client management
- **`services/embeddings.py`**: OpenAI embedding creation, contextual embeddings

### 5. Tool Files
Each tool file is a thin wrapper:
```python
# Example: src/tools/crawl_single_page.py
from src.mcp_server import mcp
from src.services.crawling import CrawlingService
from src.models import CrawlRequest, CrawlResult

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """Tool docstring"""
    service = CrawlingService(ctx)
    request = CrawlRequest(url=url)
    result = await service.crawl_single_page(request)
    return result.model_dump_json(indent=2)
```

## Migration Strategy

### Phase 1: Core Structure
1. Create directory structure
2. Move config to `config.py`
3. Create basic `models.py`
4. Set up `mcp_server.py`


### Phase 2: Services
1. Extract crawling logic to `services/crawling.py`
2. Extract search logic to `services/search.py`
3. Extract database operations to `services/database.py`
4. Extract embedding operations to `services/embeddings.py`

### Phase 3: Tools
1. Create individual tool files
2. Each tool calls appropriate service
3. Setup conftest.py for shared test setup and isntall pytest with `uv add pytest`
4. Add unit tests for each tool
5. Ensure tests pass with `uv run pytest`

### Phase 4: Utilities
1. Move text processing to `utils/text_processing.py`
2. Move reranking to `utils/reranking.py`
3. Add unit tests for each utility
4. Ensure tests pass with `uv run pytest`

### Phase 5: Cleanup
1. Remove old files
2. Update imports
3. test with `uv run pytest`


Main Issues Found

1. Monolithic File Structure

- Location: src/crawl4ai_mcp.py (1054 lines), src/utils.py (738 lines)
- Problem: Files are too large, mixing multiple concerns
  - Priority: HIGH

  2. Missing Type Safety

  - Location: Throughout the codebase
  - Problem: No Pydantic models for request/response validation
  - Example Fix:
  # src/features/crawling/models.py
  from pydantic import BaseModel, HttpUrl
  from typing import Optional, Dict, Any, List

  class CrawlRequest(BaseModel):
      url: HttpUrl
      max_depth: int = 3
      max_concurrent: int = 10
      chunk_size: int = 5000

  class CrawlResult(BaseModel):
      success: bool
      url: str
      crawl_type: str
      pages_crawled: int
      chunks_stored: int
      code_examples_stored: int = 0
      error: Optional[str] = None
  - Priority: HIGH

  3. Long Functions

  - Location:
    - crawl_single_page() - 138 lines
    - smart_crawl_url() - 194 lines
    - perform_rag_query() - 139 lines
  - Problem: Functions doing too many things, hard to test
  - Priority: HIGH

  4. Mixed Responsibilities

  - Location: utils.py
  - Problem: Contains database ops, embeddings, text processing, and more
  - Example Fix: Split into focused modules as shown in tree structure
  - Priority: MEDIUM

  5. No Configuration Validation

  - Location: Environment variable handling
  - Problem: No validation of config values
  - Example Fix:
  # src/core/config.py
  from pydantic_settings import BaseSettings
  from typing import Optional

  class Settings(BaseSettings):
      # MCP Server
      host: str = "0.0.0.0"
      port: int = 8051
      transport: str = "sse"

      # OpenAI
      openai_api_key: str
      model_choice: str = "gpt-4o-mini"

      # Features
      use_contextual_embeddings: bool = False
      use_hybrid_search: bool = False
      use_agentic_rag: bool = False
      use_reranking: bool = False

      # Supabase
      supabase_url: str
      supabase_service_key: str

      class Config:
          env_file = ".env"
  - Priority: MEDIUM

  6. Inline Business Logic in Tools

  - Location: All @mcp.tool() decorated functions
  - Problem: Business logic mixed with MCP tool interface
  - Example Fix:
  # src/features/crawling/tools/crawl_single_page.py
  from src.features.crawling.service import CrawlingService
  from src.features.crawling.models import CrawlRequest, CrawlResult

  @mcp.tool()
  async def crawl_single_page(ctx: Context, url: str) -> str:
      """Crawl a single web page and store its content."""
      service = CrawlingService(ctx.crawler, ctx.supabase_client)
      request = CrawlRequest(url=url)
      result: CrawlResult = await service.crawl_single_page(request)
      return result.model_dump_json(indent=2)
  - Priority: MEDIUM