#!/bin/bash

# PostgreSQL connection details
DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}
DB_USER=${POSTGRES_USER:-postgres}
DB_NAME=${POSTGRES_DB:-crawl4ai_rag}

echo "Setting up PostgreSQL database..."

# Create database if it doesn't exist
echo "Creating database $DB_NAME..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database already exists"

# Run the setup script
echo "Running setup script..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f setup_postgres.sql

echo "Database setup complete!" 