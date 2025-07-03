-- PostgreSQL setup script for Crawl4AI RAG MCP Server
-- Run this script as a PostgreSQL superuser to set up the database

-- Create database (uncomment if needed)
-- CREATE DATABASE crawl4ai_rag;

-- Connect to the database
\c crawl4ai_rag;

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop tables if they exist (to allow rerunning the script)
DROP TABLE IF EXISTS crawled_pages;
DROP TABLE IF EXISTS code_examples;
DROP TABLE IF EXISTS sources;

-- Create the sources table
CREATE TABLE sources ( 
    source_id text PRIMARY KEY,
    summary text,
    total_word_count integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create the documentation chunks table
CREATE TABLE crawled_pages (
    id bigserial PRIMARY KEY,
    url varchar NOT NULL,
    chunk_number integer NOT NULL,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_id text NOT NULL,
    embedding vector(1536),
    content_tokens tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    UNIQUE(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

-- Create optimized indexes for vector similarity search performance
-- IVFFLAT with optimized list count for better performance
CREATE INDEX crawled_pages_embedding_idx ON crawled_pages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Alternative: HNSW index for better accuracy and performance (comment out ivfflat if using this)
-- CREATE INDEX crawled_pages_embedding_hnsw_idx ON crawled_pages USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Create an index on metadata for faster filtering
CREATE INDEX idx_crawled_pages_metadata ON crawled_pages USING gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_crawled_pages_source_id ON crawled_pages (source_id);

-- Create composite index for source_id + similarity search optimization
CREATE INDEX idx_crawled_pages_source_embedding ON crawled_pages (source_id) INCLUDE (embedding);

-- Create a GIN index for full-text search with improved configuration
CREATE INDEX idx_crawled_pages_content_tokens ON crawled_pages USING gin (content_tokens) WITH (fastupdate = off);

-- Create index on created_at for time-based queries
CREATE INDEX idx_crawled_pages_created_at ON crawled_pages (created_at DESC);

-- Create partial index for high-quality content (similarity > 0.7)
-- CREATE INDEX idx_crawled_pages_high_quality ON crawled_pages (embedding) WHERE similarity_score > 0.7;

-- Create the code_examples table
CREATE TABLE code_examples (
    id bigserial PRIMARY KEY,
    url varchar NOT NULL,
    chunk_number integer NOT NULL,
    content text NOT NULL,  -- The code example content
    summary text NOT NULL,  -- Summary of the code example
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_id text NOT NULL,
    embedding vector(1536),
    content_tokens tsvector GENERATED ALWAYS AS (to_tsvector('english', content || ' ' || summary)) STORED,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    UNIQUE(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

-- Create optimized indexes for code examples vector search
-- IVFFLAT with optimized list count for better performance
CREATE INDEX code_examples_embedding_idx ON code_examples USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Alternative: HNSW index for better accuracy (comment out ivfflat if using this)
-- CREATE INDEX code_examples_embedding_hnsw_idx ON code_examples USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Create an index on metadata for faster filtering
CREATE INDEX idx_code_examples_metadata ON code_examples USING gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_code_examples_source_id ON code_examples (source_id);

-- Create composite index for source_id + similarity search optimization
CREATE INDEX idx_code_examples_source_embedding ON code_examples (source_id) INCLUDE (embedding);

-- Create a GIN index for full-text search on code examples with improved configuration
CREATE INDEX idx_code_examples_content_tokens ON code_examples USING gin (content_tokens) WITH (fastupdate = off);

-- Create index on created_at for time-based queries
CREATE INDEX idx_code_examples_created_at ON code_examples (created_at DESC);

-- Create a function to search for documentation chunks
CREATE OR REPLACE FUNCTION match_crawled_pages (
  query_embedding vector(1536),
  match_count int DEFAULT 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  similarity float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    crawled_pages.id,
    crawled_pages.url,
    crawled_pages.chunk_number,
    crawled_pages.content,
    crawled_pages.metadata,
    crawled_pages.source_id,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  FROM crawled_pages
  WHERE (filter = '{}'::jsonb OR crawled_pages.metadata @> filter)
    AND (source_filter IS NULL OR crawled_pages.source_id = source_filter)
  ORDER BY crawled_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create a function to search for code examples
CREATE OR REPLACE FUNCTION match_code_examples (
  query_embedding vector(1536),
  match_count int DEFAULT 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  SELECT
    code_examples.id,
    code_examples.url,
    code_examples.chunk_number,
    code_examples.content,
    code_examples.summary,
    code_examples.metadata,
    code_examples.source_id,
    1 - (code_examples.embedding <=> query_embedding) as similarity
  FROM code_examples
  WHERE (filter = '{}'::jsonb OR code_examples.metadata @> filter)
    AND (source_filter IS NULL OR code_examples.source_id = source_filter)
  ORDER BY code_examples.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create a function for hybrid search on crawled pages using RRF (Reciprocal Rank Fusion)
CREATE OR REPLACE FUNCTION hybrid_search_crawled_pages (
  query_embedding vector(1536),
  query_text text,
  match_count int DEFAULT 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL,
  rrf_k int DEFAULT 60
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  similarity float,
  text_rank float,
  hybrid_score float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  WITH vector_search AS (
    SELECT 
      crawled_pages.id,
      crawled_pages.url,
      crawled_pages.chunk_number,
      crawled_pages.content,
      crawled_pages.metadata,
      crawled_pages.source_id,
      1 - (crawled_pages.embedding <=> query_embedding) as similarity,
      ROW_NUMBER() OVER (ORDER BY crawled_pages.embedding <=> query_embedding) as vector_rank
    FROM crawled_pages
    WHERE (filter = '{}'::jsonb OR crawled_pages.metadata @> filter)
      AND (source_filter IS NULL OR crawled_pages.source_id = source_filter)
    ORDER BY crawled_pages.embedding <=> query_embedding
    LIMIT match_count * 2
  ),
  text_search AS (
    SELECT 
      crawled_pages.id,
      ts_rank_cd(crawled_pages.content_tokens, plainto_tsquery('english', query_text)) as text_rank,
      ROW_NUMBER() OVER (ORDER BY ts_rank_cd(crawled_pages.content_tokens, plainto_tsquery('english', query_text)) DESC) as text_rank_row
    FROM crawled_pages
    WHERE crawled_pages.content_tokens @@ plainto_tsquery('english', query_text)
      AND (filter = '{}'::jsonb OR crawled_pages.metadata @> filter)
      AND (source_filter IS NULL OR crawled_pages.source_id = source_filter)
    ORDER BY ts_rank_cd(crawled_pages.content_tokens, plainto_tsquery('english', query_text)) DESC
    LIMIT match_count * 2
  )
  SELECT 
    vs.id,
    vs.url,
    vs.chunk_number,
    vs.content,
    vs.metadata,
    vs.source_id,
    vs.similarity,
    COALESCE(ts.text_rank, 0.0) as text_rank,
    -- RRF formula: sum of 1/(k + rank) for each ranking
    (1.0 / (rrf_k + vs.vector_rank)) + COALESCE((1.0 / (rrf_k + ts.text_rank_row)), 0.0) as hybrid_score
  FROM vector_search vs
  LEFT JOIN text_search ts ON vs.id = ts.id
  
  UNION
  
  SELECT 
    cp.id,
    cp.url,
    cp.chunk_number,
    cp.content,
    cp.metadata,
    cp.source_id,
    COALESCE(vs.similarity, 0.0) as similarity,
    ts.text_rank,
    COALESCE((1.0 / (rrf_k + vs.vector_rank)), 0.0) + (1.0 / (rrf_k + ts.text_rank_row)) as hybrid_score
  FROM text_search ts
  JOIN crawled_pages cp ON ts.id = cp.id
  LEFT JOIN vector_search vs ON ts.id = vs.id
  WHERE vs.id IS NULL
  
  ORDER BY hybrid_score DESC
  LIMIT match_count;
END;
$$;

-- Create a function for hybrid search on code examples using RRF
CREATE OR REPLACE FUNCTION hybrid_search_code_examples (
  query_embedding vector(1536),
  query_text text,
  match_count int DEFAULT 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL,
  rrf_k int DEFAULT 60
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float,
  text_rank float,
  hybrid_score float
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  WITH vector_search AS (
    SELECT 
      code_examples.id,
      code_examples.url,
      code_examples.chunk_number,
      code_examples.content,
      code_examples.summary,
      code_examples.metadata,
      code_examples.source_id,
      1 - (code_examples.embedding <=> query_embedding) as similarity,
      ROW_NUMBER() OVER (ORDER BY code_examples.embedding <=> query_embedding) as vector_rank
    FROM code_examples
    WHERE (filter = '{}'::jsonb OR code_examples.metadata @> filter)
      AND (source_filter IS NULL OR code_examples.source_id = source_filter)
    ORDER BY code_examples.embedding <=> query_embedding
    LIMIT match_count * 2
  ),
  text_search AS (
    SELECT 
      code_examples.id,
      ts_rank_cd(code_examples.content_tokens, plainto_tsquery('english', query_text)) as text_rank,
      ROW_NUMBER() OVER (ORDER BY ts_rank_cd(code_examples.content_tokens, plainto_tsquery('english', query_text)) DESC) as text_rank_row
    FROM code_examples
    WHERE code_examples.content_tokens @@ plainto_tsquery('english', query_text)
      AND (filter = '{}'::jsonb OR code_examples.metadata @> filter)
      AND (source_filter IS NULL OR code_examples.source_id = source_filter)
    ORDER BY ts_rank_cd(code_examples.content_tokens, plainto_tsquery('english', query_text)) DESC
    LIMIT match_count * 2
  )
  SELECT 
    vs.id,
    vs.url,
    vs.chunk_number,
    vs.content,
    vs.summary,
    vs.metadata,
    vs.source_id,
    vs.similarity,
    COALESCE(ts.text_rank, 0.0) as text_rank,
    (1.0 / (rrf_k + vs.vector_rank)) + COALESCE((1.0 / (rrf_k + ts.text_rank_row)), 0.0) as hybrid_score
  FROM vector_search vs
  LEFT JOIN text_search ts ON vs.id = ts.id
  
  UNION
  
  SELECT 
    ce.id,
    ce.url,
    ce.chunk_number,
    ce.content,
    ce.summary,
    ce.metadata,
    ce.source_id,
    COALESCE(vs.similarity, 0.0) as similarity,
    ts.text_rank,
    COALESCE((1.0 / (rrf_k + vs.vector_rank)), 0.0) + (1.0 / (rrf_k + ts.text_rank_row)) as hybrid_score
  FROM text_search ts
  JOIN code_examples ce ON ts.id = ce.id
  LEFT JOIN vector_search vs ON ts.id = vs.id
  WHERE vs.id IS NULL
  
  ORDER BY hybrid_score DESC
  LIMIT match_count;
END;
$$;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_app_user;

