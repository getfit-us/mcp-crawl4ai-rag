<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [PostgreSQL with pgvector](https://github.com/pgvector/pgvector) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (PostgreSQL with pgvector), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

The server includes several advanced RAG strategies that can be enabled to enhance retrieval quality:
- **Contextual Embeddings** for enriched semantic understanding
- **True Hybrid Search** combining vector and PostgreSQL full-text search with RRF (Reciprocal Rank Fusion)
- **Query Enhancement** with query preprocessing, expansion, and multi-query generation
- **Result Diversification** using Maximum Marginal Relevance (MMR) algorithm
- **Agentic RAG** for specialized code example extraction
- **Advanced Reranking** for improved result relevance using cross-encoder models
- **Optimized Vector Storage** with enhanced indexing strategies

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

### Core Crawling & Storage
- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Enhanced Content Chunking**: Intelligently splits content with overlap preservation and boundary detection
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

### Advanced RAG Pipeline ✨ NEW
- **True Hybrid Search**: Combines vector similarity with PostgreSQL full-text search using Reciprocal Rank Fusion (RRF)
- **Query Enhancement**: Automatic query preprocessing, type detection, expansion, and multi-query generation
- **Result Diversification**: Maximum Marginal Relevance (MMR) algorithm ensures diverse, non-redundant results
- **Content-Type Aware**: Intelligently categorizes and balances different types of content (code, tutorials, API docs, etc.)
- **Optimized Vector Storage**: Enhanced indexing with IVFFLAT/HNSW options for better performance
- **Smart Query Processing**: Detects query types (factual, conceptual, code, how-to, troubleshooting) for optimized retrieval

## Tools

The server provides essential web crawling and search tools:

### Core Tools (Always Available)

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

### Conditional Tools

5. **`search_code_examples`** (requires `USE_AGENTIC_RAG=true`): Search specifically for code examples and their summaries from crawled documentation. This tool provides targeted code snippet retrieval for AI coding assistants.

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [PostgreSQL](https://www.postgresql.org/) database with [pgvector extension](https://github.com/pgvector/pgvector) (local or remote)
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)
- [NumPy](https://numpy.org/) (automatically installed with the package for enhanced MMR calculations)

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Build the Docker image:
   ```bash
   docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
   ```

3. Create a `.env` file based on the configuration section below

### Using uv directly (no Docker)

1. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

2. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   # on Mac/Linux: source .venv/bin/activate
   ```

4. Install the package:
   ```bash
   uv pip install -e .
   ```

5. Initialize Playwright (for web crawling):
   ```bash
   uv run crawl4ai-setup
   ```

5. Create a `.env` file based on the configuration section below

## Database Setup

Before running the server, you need to set up a PostgreSQL database with the pgvector extension:

1. **Install PostgreSQL and pgvector**:
   - For local setup, install PostgreSQL and the pgvector extension
   - For cloud setup, use a service that supports pgvector (e.g., Supabase, Neon, or AWS RDS with pgvector)

2. **Create the database and enable pgvector**:
   ```sql
   CREATE DATABASE crawl4ai_rag;
   \c crawl4ai_rag;
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Create the database schema**:
   - Connect to your PostgreSQL database
   - Run the SQL commands from `crawled_pages.sql` to create the necessary tables and functions

### Database Migration for Enhanced Features ✨ NEW

If you're upgrading from a previous version, you'll need to add the new full-text search columns:

```sql
-- Add full-text search columns for enhanced hybrid search
ALTER TABLE crawled_pages ADD COLUMN IF NOT EXISTS content_tokens tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
ALTER TABLE code_examples ADD COLUMN IF NOT EXISTS content_tokens tsvector GENERATED ALWAYS AS (to_tsvector('english', content || ' ' || summary)) STORED;

-- Create optimized indexes
CREATE INDEX IF NOT EXISTS idx_crawled_pages_content_tokens ON crawled_pages USING gin (content_tokens) WITH (fastupdate = off);
CREATE INDEX IF NOT EXISTS idx_code_examples_content_tokens ON code_examples USING gin (content_tokens) WITH (fastupdate = off);

-- Update existing vector indexes for better performance
DROP INDEX IF EXISTS crawled_pages_embedding_idx;
CREATE INDEX crawled_pages_embedding_idx ON crawled_pages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Add new hybrid search functions (run the full updated schema file)
```

**⚠️ Important**: Always backup your database before running migrations!

## Configuration

Create a `.env` file in the project root with the following variables:

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=streamable-http  # Options: "stdio", "sse", "streamable-http"

# OpenAI API Configuration
LLM_MODEL_API_KEY=your_openai_api_key
# Optional: Custom base URL for OpenAI-compatible endpoints (e.g., local models, Azure OpenAI)
# Use this for chat/completion models
# OPENAI_BASE_URL=https://your-custom-endpoint.com/v1
# Optional: Organization ID for OpenAI
# OPENAI_ORGANIZATION=your_org_id

# LLM for summaries and contextual embeddings
SUMMARY_LLM_MODEL=gpt-4o-mini
# Optional: Disable thinking on reasoning models (adds /no_think to prompts)
DISABLE_THINKING=false

# Embedding Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
# Optional: Custom embedding endpoint URL for custom embedding models
# If you are using a separate server for embeddings, specify its URL here
# CUSTOM_EMBEDDING_URL=https://your-embedding-api.com/embed
# Optional: Embedding service type ("openai", "huggingface", "custom")
# EMBEDDING_SERVICE_TYPE=openai

# Reranking Model Configuration
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Optional: Custom reranking model URL (e.g., from Hugging Face Hub or custom endpoint)
# CUSTOM_CROSS_ENCODER_URL=https://huggingface.co/your-username/your-reranking-model
# Optional: Local path to reranking model (takes priority over URL)
# CROSS_ENCODER_MODEL_LOCAL_PATH=/path/to/your/local/reranking/model

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false

# Enhanced RAG Pipeline Features ✨ NEW
USE_RESULT_DIVERSIFICATION=true  # Enable MMR algorithm for result diversification
MMR_LAMBDA=0.7  # Balance between relevance (1.0) and diversity (0.0)
MAX_RESULTS_PER_CONTENT_TYPE=3  # Max results per content type for diversity

# Enhanced Chunking Configuration
DEFAULT_OVERLAP=200  # Characters to overlap between chunks for context preservation

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=crawl4ai_rag
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_SSLMODE=prefer
```

### RAG Strategy Options

The Crawl4AI RAG MCP server supports four powerful RAG strategies that can be enabled independently:

#### 1. **USE_CONTEXTUAL_EMBEDDINGS**
When enabled, this strategy enhances each chunk's embedding with additional context from the entire document. The system passes both the full document and the specific chunk to an LLM (configured via `SUMMARY_LLM_MODEL`) to generate enriched context that gets embedded alongside the chunk content.

- **When to use**: Enable this when you need high-precision retrieval where context matters, such as technical documentation where terms might have different meanings in different sections.
- **Trade-offs**: Slower indexing due to LLM calls for each chunk, but significantly better retrieval accuracy.
- **Cost**: Additional LLM API calls during indexing.

#### **DISABLE_THINKING**
When enabled, this option adds `/no_think` to prompts sent to reasoning models (like OpenAI's o1-preview, o1-mini) to skip their internal thinking process and provide direct responses. This can speed up response times and reduce API costs for reasoning models when detailed thinking isn't needed.

- **When to use**: Enable this when using reasoning models for summaries and you want faster, more direct responses without the internal reasoning step.
- **Trade-offs**: Faster responses and lower costs, but may reduce the quality of reasoning for complex tasks.
- **Cost**: Lower API costs when using reasoning models as they skip the thinking step.

#### 2. **USE_HYBRID_SEARCH** ✨ ENHANCED
Combines semantic vector search with PostgreSQL full-text search using Reciprocal Rank Fusion (RRF). The system performs both searches in parallel and uses advanced ranking fusion to merge results, ensuring the best of both semantic understanding and exact keyword matching.

- **When to use**: Enable this when users might search using specific technical terms, function names, or when exact keyword matches are important alongside semantic understanding.
- **Trade-offs**: Slightly slower search queries but significantly more robust and accurate results.
- **Cost**: No additional API costs, just computational overhead.
- **New Features**: RRF algorithm, optimized full-text indexing, and intelligent result fusion.

#### 3. **USE_AGENTIC_RAG**
Enables specialized code example extraction and storage. When crawling documentation, the system identifies code blocks (≥300 characters), extracts them with surrounding context, generates summaries, and stores them in a separate vector database table specifically designed for code search.

- **When to use**: Essential for AI coding assistants that need to find specific code examples, implementation patterns, or usage examples from documentation.
- **Trade-offs**: Significantly slower crawling due to code extraction and summarization, requires more storage space.
- **Cost**: Additional LLM API calls for summarizing each code example.
- **Benefits**: Provides a dedicated `search_code_examples` tool that AI agents can use to find specific code implementations.

#### 4. **USE_RERANKING**
Applies cross-encoder reranking to search results after initial retrieval. Uses a cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) to score each result against the original query, then reorders results by relevance. Supports custom models via URLs or local paths.

- **When to use**: Enable this when search precision is critical and you need the most relevant results at the top. Particularly useful for complex queries where semantic similarity alone might not capture query intent.
- **Trade-offs**: Adds ~100-200ms to search queries depending on result count, but significantly improves result ordering.
- **Cost**: No additional API costs - uses a local model that runs on CPU.
- **Benefits**: Better result relevance, especially for complex queries. Works with both regular RAG search and code example search.
- **Custom Models**: You can use custom reranking models from Hugging Face Hub, local files, or custom endpoints.

### Enhanced RAG Pipeline Features ✨ NEW

The system now includes several advanced features that significantly improve retrieval accuracy and result quality:

#### 5. **USE_RESULT_DIVERSIFICATION**
Applies Maximum Marginal Relevance (MMR) algorithm to ensure diverse, non-redundant search results. The system balances relevance with diversity to prevent similar results from dominating the response.

- **When to use**: Always recommended. Ensures comprehensive coverage of topics and reduces redundancy in search results.
- **Configuration**: 
  - `MMR_LAMBDA=0.7` - Balance between relevance (1.0) and diversity (0.0)
  - `MAX_RESULTS_PER_CONTENT_TYPE=3` - Limits results per content type for variety
- **Trade-offs**: Minimal performance impact with significant improvement in result quality.
- **Benefits**: More comprehensive answers, better coverage of different aspects of a topic.

#### 6. **Enhanced Query Processing** (Always Active)
Automatically enhances queries through multiple strategies:

- **Query Type Detection**: Identifies query intent (factual, conceptual, code, how-to, troubleshooting)
- **Query Expansion**: Generates variations based on detected type
- **Multi-Query Generation**: Creates alternative phrasings for comprehensive retrieval
- **Technical Term Preservation**: Maintains proper capitalization for technical terms

- **Benefits**: Better understanding of user intent, more comprehensive retrieval, improved accuracy.
- **Trade-offs**: Slightly longer processing time for significantly better results.

#### 7. **Enhanced Content Chunking** (Always Active)
Improved chunking strategy with overlap preservation:

- **Intelligent Overlap**: `DEFAULT_OVERLAP=200` characters preserved between chunks
- **Boundary Detection**: Respects code blocks, paragraphs, and sentence boundaries
- **Context Preservation**: Prevents information loss at chunk boundaries

- **Benefits**: Better context preservation, improved retrieval accuracy, reduced information fragmentation.

#### 8. **Optimized Vector Storage** (Always Active)
Enhanced database indexing and query optimization:

- **IVFFLAT Optimization**: Tuned list counts for better performance
- **HNSW Support**: Alternative indexing for higher accuracy (commented in schema)
- **Composite Indexes**: Source + embedding optimization for filtered searches
- **Full-Text Optimization**: Improved GIN indexing configuration

- **Benefits**: Faster search queries, better scalability, improved accuracy.

### Batch Processing Configuration

The MCP server supports batch processing for embeddings and summarization to significantly speed up crawling operations when dealing with multiple documents or chunks:

#### **ENABLE_BATCH_EMBEDDINGS** (default: true)
Enables batch processing for embedding generation, processing multiple texts in a single API call.

- **EMBEDDING_BATCH_SIZE** (default: 100): Number of texts to process in each batch. Larger batches are more efficient but may hit API limits.
- **When to use**: Keep enabled for faster embedding generation. Only disable for debugging individual embedding issues.
- **Benefits**: Can provide 3-10x speedup when processing multiple chunks.

#### **ENABLE_BATCH_SUMMARIES** (default: false)
Enables batch processing for LLM-based summarization (code examples and source summaries).

- **SUMMARY_BATCH_SIZE** (default: 10): Number of summaries to generate in parallel. Balance between speed and API rate limits.
- **When to use**: Enable when crawling multiple pages or documents with many code examples.
- **Trade-offs**: Faster processing but may hit LLM API rate limits with large batches.
- **Benefits**: Can provide 5-15x speedup for summary generation.

#### **ENABLE_BATCH_CONTEXTUAL_EMBEDDINGS** (default: false)
Enables batch processing for contextual embedding generation (when USE_CONTEXTUAL_EMBEDDINGS is enabled).

- **CONTEXTUAL_EMBEDDING_BATCH_SIZE** (default: 20): Number of contextual embeddings to process in parallel.
- **When to use**: Enable when using contextual embeddings on documents with many chunks.
- **Trade-offs**: Faster contextual processing but higher LLM API usage.
- **Benefits**: Can provide 5-10x speedup for contextual embedding generation.

### Recommended Configurations

**For general documentation RAG (Enhanced):**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
USE_RESULT_DIVERSIFICATION=true
MMR_LAMBDA=0.7
DEFAULT_OVERLAP=200
```

**For AI coding assistant with code examples (Enhanced):**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_RESULT_DIVERSIFICATION=true
MMR_LAMBDA=0.6  # Slightly more diversity for code examples
MAX_RESULTS_PER_CONTENT_TYPE=2
DEFAULT_OVERLAP=300  # More overlap for code context
ENABLE_BATCH_EMBEDDINGS=true
ENABLE_BATCH_SUMMARIES=true
ENABLE_BATCH_CONTEXTUAL_EMBEDDINGS=true
```

**For fast, basic RAG (Enhanced):**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_RESULT_DIVERSIFICATION=true
MMR_LAMBDA=0.8  # Favor relevance over diversity for speed
DEFAULT_OVERLAP=150
ENABLE_BATCH_EMBEDDINGS=true
```

**For maximum processing speed (with higher API usage):**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
ENABLE_BATCH_EMBEDDINGS=true
EMBEDDING_BATCH_SIZE=200
ENABLE_BATCH_SUMMARIES=true
SUMMARY_BATCH_SIZE=20
ENABLE_BATCH_CONTEXTUAL_EMBEDDINGS=true
CONTEXTUAL_EMBEDDING_BATCH_SIZE=50
```

## Custom Model Configuration

### Custom Reranking Models

The system supports custom reranking models through several configuration options:

#### 1. **Hugging Face Hub Models**
You can use any cross-encoder model from Hugging Face Hub:

```bash
# Use a different model from Hugging Face
CROSS_ENCODER_MODEL=sentence-transformers/ms-marco-MiniLM-L-12-v2

# Or specify a custom model via URL
CUSTOM_CROSS_ENCODER_URL=https://huggingface.co/your-username/your-custom-reranker
```

#### 2. **Local Model Files**
For offline usage or custom trained models:

```bash
# Use a local model directory
CROSS_ENCODER_MODEL_LOCAL_PATH=/path/to/your/local/reranking/model
```

#### 3. **Priority Order**
The system checks for models in this order:
1. **Local Path** (`CROSS_ENCODER_MODEL_LOCAL_PATH`) - highest priority
2. **Custom URL** (`CUSTOM_CROSS_ENCODER_URL`) - second priority  
3. **Default Model** (`CROSS_ENCODER_MODEL`) - fallback

#### 4. **Popular Reranking Models**
Some recommended cross-encoder models:

```bash
# Lightweight and fast (default)
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# More accurate but slower
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2

# Multilingual support
CROSS_ENCODER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# Domain-specific (e.g., for biomedical content)
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-electra-base
```

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run crawl4ai-mcp
```

The server will start and listen on the configured host and port.

## Troubleshooting

### `ConnectionRefusedError`

If you encounter a `ConnectionRefusedError` when starting the server, it means the application cannot connect to your PostgreSQL database. This is typically a configuration issue.

- **Check `POSTGRES_HOST`**: If your database is running on the same machine as the application, ensure `POSTGRES_HOST` in your `.env` file is set to `localhost`, not an IP address like `192.168.1.244`. Many default PostgreSQL installations only listen for connections on `localhost` (`127.0.0.1`).

- **Check if PostgreSQL is running**: Make sure your PostgreSQL server is active.

- **Check PostgreSQL `listen_addresses`**: If you must use an IP address, ensure your PostgreSQL server is configured to listen on that address. You may need to edit your `postgresql.conf` file and set `listen_addresses = '*'`.

## Integration with MCP Clients

### Transport Options

The MCP server supports three transport methods:

#### 1. Streamable HTTP Configuration (Recommended for MCP 1.9.4+)

For the latest streamable HTTP transport support:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "streamable-http",
      "url": "http://localhost:8051"
    }
  }
}
```

Set `TRANSPORT=streamable-http` in your `.env` file and start the server normally.

#### 2. SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

#### 3. Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "uv",
      "args": ["run", "crawl4ai-mcp"],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_MODEL_API_KEY": "your_openai_api_key",
        "OPENAI_BASE_URL": "https://your-custom-endpoint.com/v1",
        "OPENAI_ORGANIZATION": "your_org_id",
        "SUMMARY_LLM_MODEL": "gpt-4o-mini",
        "DISABLE_THINKING": "false",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": "1536",
        "CUSTOM_EMBEDDING_URL": "https://your-embedding-api.com/embed",
        "EMBEDDING_SERVICE_TYPE": "openai",
        "CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "CUSTOM_CROSS_ENCODER_URL": "https://huggingface.co/your-username/your-reranking-model",
        "CROSS_ENCODER_MODEL_LOCAL_PATH": "/path/to/your/local/reranking/model",
        "USE_CONTEXTUAL_EMBEDDINGS": "false",
        "USE_HYBRID_SEARCH": "true",
        "USE_AGENTIC_RAG": "false",
        "USE_RERANKING": "true",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "crawl4ai_rag",
        "POSTGRES_USER": "your_postgres_user",
        "POSTGRES_PASSWORD": "your_postgres_password",
        "POSTGRES_SSLMODE": "prefer"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "LLM_MODEL_API_KEY", 
               "-e", "OPENAI_BASE_URL",
               "-e", "OPENAI_ORGANIZATION",
               "-e", "SUMMARY_LLM_MODEL",
               "-e", "DISABLE_THINKING",
               "-e", "EMBEDDING_MODEL",
               "-e", "EMBEDDING_DIMENSIONS",
               "-e", "CUSTOM_EMBEDDING_URL",
               "-e", "EMBEDDING_SERVICE_TYPE",
               "-e", "CROSS_ENCODER_MODEL",
               "-e", "CUSTOM_CROSS_ENCODER_URL",
               "-e", "CROSS_ENCODER_MODEL_LOCAL_PATH",
               "-e", "USE_CONTEXTUAL_EMBEDDINGS",
               "-e", "USE_HYBRID_SEARCH",
               "-e", "USE_AGENTIC_RAG",
               "-e", "USE_RERANKING",
               "-e", "ENABLE_BATCH_EMBEDDINGS",
               "-e", "EMBEDDING_BATCH_SIZE",
               "-e", "ENABLE_BATCH_SUMMARIES", 
               "-e", "SUMMARY_BATCH_SIZE",
               "-e", "ENABLE_BATCH_CONTEXTUAL_EMBEDDINGS",
               "-e", "CONTEXTUAL_EMBEDDING_BATCH_SIZE",
               "-e", "POSTGRES_HOST", 
               "-e", "POSTGRES_PORT", 
               "-e", "POSTGRES_DB", 
               "-e", "POSTGRES_USER", 
               "-e", "POSTGRES_PASSWORD", 
               "-e", "POSTGRES_SSLMODE", 
               "mcp/crawl4ai-rag"],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_MODEL_API_KEY": "your_openai_api_key",
        "OPENAI_BASE_URL": "https://your-custom-endpoint.com/v1",
        "OPENAI_ORGANIZATION": "your_org_id",
        "SUMMARY_LLM_MODEL": "gpt-4o-mini",
        "DISABLE_THINKING": "false",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": "1536",
        "CUSTOM_EMBEDDING_URL": "https://your-embedding-api.com/embed",
        "EMBEDDING_SERVICE_TYPE": "openai",
        "CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "CUSTOM_CROSS_ENCODER_URL": "https://huggingface.co/your-username/your-reranking-model",
        "CROSS_ENCODER_MODEL_LOCAL_PATH": "/path/to/your/local/reranking/model",
        "USE_CONTEXTUAL_EMBEDDINGS": "false",
        "USE_HYBRID_SEARCH": "true",
        "USE_AGENTIC_RAG": "false",
        "USE_RERANKING": "true",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "crawl4ai_rag",
        "POSTGRES_USER": "your_postgres_user",
        "POSTGRES_PASSWORD": "your_postgres_password",
        "POSTGRES_SSLMODE": "prefer"
      }
    }
  }
}
```

## PostgreSQL Setup Examples

### Local PostgreSQL Setup

1. **Install PostgreSQL and pgvector**:
   ```bash
   # On macOS with Homebrew
   brew install postgresql pgvector
   
   # On Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   # Install pgvector from source or package manager
   
   # On Docker
   docker run --name postgres-pgvector \
     -e POSTGRES_PASSWORD=mypassword \
     -e POSTGRES_DB=crawl4ai_rag \
     -p 5432:5432 \
     -d pgvector/pgvector:pg16
   ```

2. **Set up the database**:
   ```bash
   # Connect to PostgreSQL
   psql -h localhost -U postgres
   
   # Run the setup script
   \i setup_postgres.sql
   ```

3. **Create application user** (optional but recommended):
   ```sql
   CREATE USER crawl4ai_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE crawl4ai_rag TO crawl4ai_user;
   \c crawl4ai_rag;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO crawl4ai_user;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO crawl4ai_user;
   GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO crawl4ai_user;
   ```

### Cloud PostgreSQL Setup

You can use any cloud PostgreSQL service that supports pgvector:

- **Supabase**: Has pgvector enabled by default
- **Neon**: Supports pgvector extension
- **AWS RDS**: Supports pgvector (PostgreSQL 15+)
- **Google Cloud SQL**: Supports pgvector
- **Azure Database for PostgreSQL**: Supports pgvector

Simply run the `setup_postgres.sql` script on your cloud database after enabling the pgvector extension.

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers

## Migration from Supabase

If you're migrating from the original Supabase version:

1. Export your data from Supabase
2. Set up PostgreSQL with pgvector using the instructions above
3. Import your data to the new PostgreSQL database
4. Update your configuration to use PostgreSQL connection settings
5. The MCP tools and API remain the same