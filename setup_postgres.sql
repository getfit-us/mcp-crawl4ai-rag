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
    embedding vector(1024),
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    UNIQUE(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

-- Create an index for better vector similarity search performance
CREATE INDEX crawled_pages_embedding_idx ON crawled_pages USING ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
CREATE INDEX idx_crawled_pages_metadata ON crawled_pages USING gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_crawled_pages_source_id ON crawled_pages (source_id);

-- Create the code_examples table
CREATE TABLE code_examples (
    id bigserial PRIMARY KEY,
    url varchar NOT NULL,
    chunk_number integer NOT NULL,
    content text NOT NULL,  -- The code example content
    summary text NOT NULL,  -- Summary of the code example
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_id text NOT NULL,
    embedding vector(1024),
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    UNIQUE(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    FOREIGN KEY (source_id) REFERENCES sources(source_id)
);

-- Create an index for better vector similarity search performance
CREATE INDEX code_examples_embedding_idx ON code_examples USING ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
CREATE INDEX idx_code_examples_metadata ON code_examples USING gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_code_examples_source_id ON code_examples (source_id);

-- Create a function to search for documentation chunks
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
  WHERE (filter = '{}'::jsonb OR crawled_pages.metadata @> filter)
    AND (source_filter IS NULL OR crawled_pages.source_id = source_filter)
  ORDER BY crawled_pages.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create a function to search for code examples
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
  WHERE (filter = '{}'::jsonb OR code_examples.metadata @> filter)
    AND (source_filter IS NULL OR code_examples.source_id = source_filter)
  ORDER BY code_examples.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Grant necessary permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_app_user;

