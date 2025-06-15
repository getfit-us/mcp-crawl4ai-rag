# PostgreSQL PGVector Tests

This directory contains tests for the DatabaseService that use a real local PostgreSQL database with PGVector extension instead of mocked Supabase clients.

## Prerequisites

1. **PostgreSQL 14+** with **PGVector extension** installed
2. **Python dependencies**: `asyncpg`, `pytest`, `pytest-asyncio`

## Setup

### 1. Install PostgreSQL and PGVector

**macOS (using Homebrew):**
```bash
brew install postgresql pgvector
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-17-pgvector  # or appropriate version
sudo systemctl start postgresql
```

**Docker (alternative):**
```bash
docker run --name test-postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=test_crawl4ai \
  -p 5432:5432 \
  -d ankane/pgvector
```

### 2. Create Test Database

Connect to PostgreSQL as superuser and create the test database:

```bash
# Connect to PostgreSQL
psql -U postgres

# Create test database
CREATE DATABASE test_crawl4ai;

# Exit psql
\q
```

### 3. Setup Database Schema

Run the setup script to create tables and enable extensions:

```bash
# From the project root directory
psql -U postgres -d test_crawl4ai -f setup_test_postgres.sql
```

### 4. Configure Connection

The tests expect the following database configuration:
- **Host**: localhost
- **Port**: 5432  
- **Database**: test_crawl4ai
- **Username**: postgres
- **Password**: password

You can modify the connection string in `test_database_postgres.py` if your setup is different:

```python
TEST_DATABASE_URL = "postgresql://postgres:password@localhost:5432/test_crawl4ai"
```

## Running Tests

### Install Python Dependencies

```bash
pip install pytest pytest-asyncio asyncpg
```

### Run All PostgreSQL Tests

```bash
# From the project root directory
pytest src/services/tests/test_database_postgres.py -v
```

### Run Specific Tests

```bash
# Test document addition
pytest src/services/tests/test_database_postgres.py::test_add_documents_success -v

# Test code examples
pytest src/services/tests/test_database_postgres.py::test_add_code_examples_success -v

# Test vector operations
pytest src/services/tests/test_database_postgres.py::test_database_with_real_vector_operations -v
```

### Run with Coverage

```bash
pytest src/services/tests/test_database_postgres.py --cov=src/services/database --cov-report=html
```

## Test Features

The PostgreSQL tests include:

- **Real Database Operations**: Tests interact with an actual PostgreSQL database
- **PGVector Integration**: Tests verify that vector embeddings are stored and retrieved correctly
- **Transaction Safety**: Each test is isolated with database cleanup
- **Batch Operations**: Tests verify batch insertion and processing
- **Error Handling**: Tests cover error scenarios and edge cases
- **Performance**: Tests include batch processing and multiple document scenarios

## Test Structure

### Fixtures

- `postgres_pool`: Session-scoped PostgreSQL connection pool
- `clean_database`: Function-scoped database cleanup
- `database_service`: DatabaseService instance with real PostgreSQL connection
- `test_settings`: Mock settings for testing

### Key Test Categories

1. **Document Management**
   - Adding single and multiple documents
   - Batch processing
   - Document replacement
   - Empty list handling

2. **Code Examples**
   - Adding code examples with embeddings
   - Language detection
   - Metadata handling

3. **Source Management**
   - Creating new sources
   - Updating existing sources
   - Source information retrieval

4. **Vector Operations**
   - Embedding storage and retrieval
   - Vector dimension validation
   - PGVector functionality

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure PostgreSQL is running
   - Check connection parameters
   - Verify database exists

2. **PGVector Extension Not Found**
   - Install pgvector extension
   - Run `CREATE EXTENSION vector;` in the database

3. **Permission Denied**
   - Check user permissions
   - Grant necessary privileges on tables

4. **Test Database Already Exists**
   - Drop and recreate: `DROP DATABASE test_crawl4ai; CREATE DATABASE test_crawl4ai;`

### Debugging

Enable debug logging by setting environment variable:
```bash
export PYTEST_CURRENT_TEST=1
pytest src/services/tests/test_database_postgres.py -v -s --log-level=DEBUG
```

## Cleanup

To clean up test data and database:

```bash
# Connect to PostgreSQL
psql -U postgres

# Drop test database
DROP DATABASE test_crawl4ai;

# Exit
\q
```

For Docker:
```bash
docker stop test-postgres
docker rm test-postgres
``` 