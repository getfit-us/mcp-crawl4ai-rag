#!/usr/bin/env python3
"""Script to set up the PostgreSQL test database."""

import asyncio
import asyncpg
import sys
from dotenv import load_dotenv
import os

load_dotenv()


async def setup_test_database():
    """Set up the test database with required schema."""
    
    # Database configuration
    DB_CONFIG = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT"),
        "user": os.getenv("POSTGRES_USER") ,
        "password": os.getenv("POSTGRES_PASSWORD")
    }
    
    TEST_DB_NAME = "test_crawl4ai"
    
    try:
        print("Connecting to PostgreSQL...")
        # Connect to default postgres database first
        conn = await asyncpg.connect(database="postgres", **DB_CONFIG)
        
        # Check if test database exists
        db_exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB_NAME
        )
        
        if db_exists:
            print(f"Database '{TEST_DB_NAME}' already exists. Dropping and recreating...")
            # Terminate existing connections
            await conn.execute(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{TEST_DB_NAME}'"
            )
            await conn.execute(f"DROP DATABASE {TEST_DB_NAME}")
        
        # Create test database
        print(f"Creating database '{TEST_DB_NAME}'...")
        await conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
        await conn.close()
        
        # Connect to test database and set up schema
        print("Setting up database schema...")
        test_conn = await asyncpg.connect(database=TEST_DB_NAME, **DB_CONFIG)
        
        # Execute schema setup manually to avoid SQL parsing issues
        print("Setting up schema manually...")
        
        # Enable pgvector extension
        await test_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create sources table (matching setup_postgres.sql exactly)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS sources ( 
                source_id text PRIMARY KEY,
                summary text,
                total_word_count integer DEFAULT 0,
                created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
                updated_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
            )
        """)
        
        # Create crawled_pages table (matching setup_postgres.sql exactly)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS crawled_pages (
                id bigserial PRIMARY KEY,
                url varchar NOT NULL,
                chunk_number integer NOT NULL,
                content text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                source_id text NOT NULL,
                embedding vector(1536),
                created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
                
                UNIQUE(url, chunk_number),
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
        """)
        
        # Create code_examples table (matching setup_postgres.sql exactly)
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS code_examples (
                id bigserial PRIMARY KEY,
                url varchar NOT NULL,
                chunk_number integer NOT NULL,
                content text NOT NULL,
                summary text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                source_id text NOT NULL,
                embedding vector(1536),
                created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
                
                UNIQUE(url, chunk_number),
                FOREIGN KEY (source_id) REFERENCES sources(source_id)
            )
        """)
        
        # Create indexes (matching setup_postgres.sql exactly)
        await test_conn.execute("CREATE INDEX IF NOT EXISTS crawled_pages_embedding_idx ON crawled_pages USING ivfflat (embedding vector_cosine_ops)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_metadata ON crawled_pages USING gin (metadata)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_pages_source_id ON crawled_pages (source_id)")
        
        await test_conn.execute("CREATE INDEX IF NOT EXISTS code_examples_embedding_idx ON code_examples USING ivfflat (embedding vector_cosine_ops)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_code_examples_metadata ON code_examples USING gin (metadata)")
        await test_conn.execute("CREATE INDEX IF NOT EXISTS idx_code_examples_source_id ON code_examples (source_id)")
        
        # Create search functions
        await test_conn.execute("""
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
        """)
        
        await test_conn.execute("""
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
        """)
        
        await test_conn.close()
        
        print(f"✅ Test database '{TEST_DB_NAME}' setup completed successfully!")
        print(f"Connection string: postgresql://postgres:password@localhost:5432/{TEST_DB_NAME}")
        print("\nYou can now run the tests with:")
        print("pytest src/services/tests/test_database_postgres.py -v")
        
    except asyncpg.InvalidCatalogNameError:
        print("❌ Error: Could not connect to PostgreSQL. Make sure:")
        print("  1. PostgreSQL is running")
        print("  2. User 'postgres' exists with password 'password'")
        print("  3. Connection settings are correct")
        sys.exit(1)
    except asyncpg.InsufficientPrivilegeError:
        print("❌ Error: Insufficient privileges. Make sure the postgres user has:")
        print("  1. CREATEDB privilege")
        print("  2. Superuser privileges (for creating extensions)")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error setting up test database: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    print("PostgreSQL Test Database Setup")
    print("=" * 40)
    print()
    
    try:
        asyncio.run(setup_test_database())
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main() 