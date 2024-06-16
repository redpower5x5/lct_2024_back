#!/bin/bash
until pg_isready -h localhost -p 5432 -U postgres; do
  echo "Waiting for PostgreSQL to start..."
  sleep 2
done

# Create database and enable pgvector extension
psql -U postgres -c "CREATE DATABASE videodb;"
# Initialize the PostgreSQL database and enable pgvector extension
psql -U postgres -d videodb -c "CREATE EXTENSION IF NOT EXISTS vector;"
