-- Enable the pgvector extension
create extension if not exists vector;

-- Drop tables if they exist (to allow rerunning the script)
drop table if exists crawled_pages;
drop table if exists code_examples;
drop table if exists sources;

-- Create the sources table
create table sources (
    source_id text primary key,
    summary text,
    total_word_count integer default 0,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create the documentation chunks table
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    content_tokens tsvector generated always as (to_tsvector('english', content)) stored,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    foreign key (source_id) references sources(source_id)
);

-- Create optimized indexes for vector similarity search performance
-- IVFFLAT with optimized list count for better performance
create index crawled_pages_embedding_idx on crawled_pages using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Alternative: HNSW index for better accuracy and performance (comment out ivfflat if using this)
-- create index crawled_pages_embedding_hnsw_idx on crawled_pages using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);

-- Create an index on metadata for faster filtering
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_crawled_pages_source_id ON crawled_pages (source_id);

-- Create composite index for source_id + similarity search optimization
create index idx_crawled_pages_source_embedding on crawled_pages (source_id) include (embedding);

-- Create a GIN index for full-text search with improved configuration
create index idx_crawled_pages_content_tokens on crawled_pages using gin (content_tokens) with (fastupdate = off);

-- Create index on created_at for time-based queries
create index idx_crawled_pages_created_at on crawled_pages (created_at desc);

-- Create a function to search for documentation chunks
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    metadata,
    source_id,
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  order by crawled_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the crawled_pages table
alter table crawled_pages enable row level security;

-- Create a policy that allows anyone to read crawled_pages
create policy "Allow public read access to crawled_pages"
  on crawled_pages
  for select
  to public
  using (true);

-- Enable RLS on the sources table
alter table sources enable row level security;

-- Create a policy that allows anyone to read sources
create policy "Allow public read access to sources"
  on sources
  for select
  to public
  using (true);

-- Create the code_examples table
create table code_examples (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- The code example content
    summary text not null,  -- Summary of the code example
    metadata jsonb not null default '{}'::jsonb,
    source_id text not null,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    content_tokens tsvector generated always as (to_tsvector('english', content || ' ' || summary)) stored,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number),
    
    -- Add foreign key constraint to sources table
    foreign key (source_id) references sources(source_id)
);

-- Create optimized indexes for code examples vector search
-- IVFFLAT with optimized list count for better performance
create index code_examples_embedding_idx on code_examples using ivfflat (embedding vector_cosine_ops) with (lists = 50);

-- Alternative: HNSW index for better accuracy (comment out ivfflat if using this)
-- create index code_examples_embedding_hnsw_idx on code_examples using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);

-- Create an index on metadata for faster filtering
create index idx_code_examples_metadata on code_examples using gin (metadata);

-- Create an index on source_id for faster filtering
CREATE INDEX idx_code_examples_source_id ON code_examples (source_id);

-- Create composite index for source_id + similarity search optimization
create index idx_code_examples_source_embedding on code_examples (source_id) include (embedding);

-- Create a GIN index for full-text search on code examples with improved configuration
create index idx_code_examples_content_tokens on code_examples using gin (content_tokens) with (fastupdate = off);

-- Create index on created_at for time-based queries
create index idx_code_examples_created_at on code_examples (created_at desc);

-- Create a function to search for code examples
create or replace function match_code_examples (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb,
  source_filter text DEFAULT NULL
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  summary text,
  metadata jsonb,
  source_id text,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    summary,
    metadata,
    source_id,
    1 - (code_examples.embedding <=> query_embedding) as similarity
  from code_examples
  where metadata @> filter
    AND (source_filter IS NULL OR source_id = source_filter)
  order by code_examples.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the code_examples table
alter table code_examples enable row level security;

-- Create a policy that allows anyone to read code_examples
create policy "Allow public read access to code_examples"
  on code_examples
  for select
  to public
  using (true);

-- Create a function for hybrid search on crawled pages using RRF (Reciprocal Rank Fusion)
create or replace function hybrid_search_crawled_pages (
  query_embedding vector(1536),
  query_text text,
  match_count int default 10,
  filter jsonb default '{}'::jsonb,
  source_filter text default null,
  rrf_k int default 60
) returns table (
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
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  with vector_search as (
    select 
      crawled_pages.id,
      crawled_pages.url,
      crawled_pages.chunk_number,
      crawled_pages.content,
      crawled_pages.metadata,
      crawled_pages.source_id,
      1 - (crawled_pages.embedding <=> query_embedding) as similarity,
      row_number() over (order by crawled_pages.embedding <=> query_embedding) as vector_rank
    from crawled_pages
    where (filter = '{}'::jsonb or crawled_pages.metadata @> filter)
      and (source_filter is null or crawled_pages.source_id = source_filter)
    order by crawled_pages.embedding <=> query_embedding
    limit match_count * 2
  ),
  text_search as (
    select 
      crawled_pages.id,
      ts_rank_cd(crawled_pages.content_tokens, plainto_tsquery('english', query_text)) as text_rank,
      row_number() over (order by ts_rank_cd(crawled_pages.content_tokens, plainto_tsquery('english', query_text)) desc) as text_rank_row
    from crawled_pages
    where crawled_pages.content_tokens @@ plainto_tsquery('english', query_text)
      and (filter = '{}'::jsonb or crawled_pages.metadata @> filter)
      and (source_filter is null or crawled_pages.source_id = source_filter)
    order by ts_rank_cd(crawled_pages.content_tokens, plainto_tsquery('english', query_text)) desc
    limit match_count * 2
  )
  select 
    vs.id,
    vs.url,
    vs.chunk_number,
    vs.content,
    vs.metadata,
    vs.source_id,
    vs.similarity,
    coalesce(ts.text_rank, 0.0) as text_rank,
    -- RRF formula: sum of 1/(k + rank) for each ranking
    (1.0 / (rrf_k + vs.vector_rank)) + coalesce((1.0 / (rrf_k + ts.text_rank_row)), 0.0) as hybrid_score
  from vector_search vs
  left join text_search ts on vs.id = ts.id
  
  union
  
  select 
    cp.id,
    cp.url,
    cp.chunk_number,
    cp.content,
    cp.metadata,
    cp.source_id,
    coalesce(vs.similarity, 0.0) as similarity,
    ts.text_rank,
    coalesce((1.0 / (rrf_k + vs.vector_rank)), 0.0) + (1.0 / (rrf_k + ts.text_rank_row)) as hybrid_score
  from text_search ts
  join crawled_pages cp on ts.id = cp.id
  left join vector_search vs on ts.id = vs.id
  where vs.id is null
  
  order by hybrid_score desc
  limit match_count;
end;
$$;

-- Create a function for hybrid search on code examples using RRF
create or replace function hybrid_search_code_examples (
  query_embedding vector(1536),
  query_text text,
  match_count int default 10,
  filter jsonb default '{}'::jsonb,
  source_filter text default null,
  rrf_k int default 60
) returns table (
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
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  with vector_search as (
    select 
      code_examples.id,
      code_examples.url,
      code_examples.chunk_number,
      code_examples.content,
      code_examples.summary,
      code_examples.metadata,
      code_examples.source_id,
      1 - (code_examples.embedding <=> query_embedding) as similarity,
      row_number() over (order by code_examples.embedding <=> query_embedding) as vector_rank
    from code_examples
    where (filter = '{}'::jsonb or code_examples.metadata @> filter)
      and (source_filter is null or code_examples.source_id = source_filter)
    order by code_examples.embedding <=> query_embedding
    limit match_count * 2
  ),
  text_search as (
    select 
      code_examples.id,
      ts_rank_cd(code_examples.content_tokens, plainto_tsquery('english', query_text)) as text_rank,
      row_number() over (order by ts_rank_cd(code_examples.content_tokens, plainto_tsquery('english', query_text)) desc) as text_rank_row
    from code_examples
    where code_examples.content_tokens @@ plainto_tsquery('english', query_text)
      and (filter = '{}'::jsonb or code_examples.metadata @> filter)
      and (source_filter is null or code_examples.source_id = source_filter)
    order by ts_rank_cd(code_examples.content_tokens, plainto_tsquery('english', query_text)) desc
    limit match_count * 2
  )
  select 
    vs.id,
    vs.url,
    vs.chunk_number,
    vs.content,
    vs.summary,
    vs.metadata,
    vs.source_id,
    vs.similarity,
    coalesce(ts.text_rank, 0.0) as text_rank,
    (1.0 / (rrf_k + vs.vector_rank)) + coalesce((1.0 / (rrf_k + ts.text_rank_row)), 0.0) as hybrid_score
  from vector_search vs
  left join text_search ts on vs.id = ts.id
  
  union
  
  select 
    ce.id,
    ce.url,
    ce.chunk_number,
    ce.content,
    ce.summary,
    ce.metadata,
    ce.source_id,
    coalesce(vs.similarity, 0.0) as similarity,
    ts.text_rank,
    coalesce((1.0 / (rrf_k + vs.vector_rank)), 0.0) + (1.0 / (rrf_k + ts.text_rank_row)) as hybrid_score
  from text_search ts
  join code_examples ce on ts.id = ce.id
  left join vector_search vs on ts.id = vs.id
  where vs.id is null
  
  order by hybrid_score desc
  limit match_count;
end;
$$;