-- Setup script for test PostgreSQL database with PGVector
-- Run this script to prepare your local PostgreSQL instance for testing

-- Create test database (run this as superuser)
-- DROP DATABASE IF EXISTS test_crawl4ai;
-- CREATE DATABASE test_crawl4ai;

-- Connect to test_crawl4ai database and run the following:

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create sources table
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    summary TEXT,
    total_word_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create crawled_pages table with pgvector support
CREATE TABLE IF NOT EXISTS crawled_pages (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    source_id TEXT,
    embedding vector(1024),  -- PostgreSQL embedding dimensions
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

-- Create code_examples table with pgvector support
CREATE TABLE IF NOT EXISTS code_examples (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    metadata JSONB,
    source_id TEXT,
    embedding vector(1024),  -- PostgreSQL embedding dimensions
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_crawled_pages_url ON crawled_pages(url);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_source_id ON crawled_pages(source_id);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_embedding ON crawled_pages USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_code_examples_url ON code_examples(url);
CREATE INDEX IF NOT EXISTS idx_code_examples_source_id ON code_examples(source_id);
CREATE INDEX IF NOT EXISTS idx_code_examples_embedding ON code_examples USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_sources_source_id ON sources(source_id);

-- Create search functions

-- Function to search for documentation chunks
CREATE OR REPLACE FUNCTION match_crawled_pages (
  query_embedding vector(1024),
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
  WHERE crawled_pages.metadata @> filter
    AND (source_filter IS NULL OR crawled_pages.source_id = source_filter)
  ORDER BY crawled_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Function to search for code examples
CREATE OR REPLACE FUNCTION match_code_examples (
  query_embedding vector(1024),
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
  WHERE code_examples.metadata @> filter
    AND (source_filter IS NULL OR code_examples.source_id = source_filter)
  ORDER BY code_examples.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO postgres; 