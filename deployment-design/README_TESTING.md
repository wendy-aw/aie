# Testing Guide for Deployment Design

This guide explains how to run the tests for the deployment-design components.

## Test Structure

The testing suite consists of two main components:

1. **Elastic Backend Unit Tests** - Tests for the indexing functionality
2. **Integration Tests** - End-to-end tests for the complete system

## Running Tests

All tests are now organized in the `tests/` directory for easy discovery.

### All Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_cv_index.py -v        # Unit tests only
pytest tests/test_integration.py -v     # Integration tests only
```

### Integration Tests

Integration tests require the full stack to be running:

```bash
# Start the complete platform
docker compose up --build -d

# Run integration tests
pytest tests/test_integration.py -v

# Cleanup
docker compose down
```

## Test Coverage

### Elastic Backend Tests (`test_cv_index.py`)

- Elasticsearch client connection and retry logic
- Index mapping creation and configuration
- Index creation and deletion
- CSV data processing and validation
- Document generation for bulk indexing
- Bulk indexing operations and error handling
- Index verification after indexing


### Integration Tests (`test_integration.py`)

- Elasticsearch cluster health and node status
- Index creation with proper mapping
- Data insertion and document counting
- Search functionality including basic and faceted search
- Range queries on numeric fields (duration)
- Search UI accessibility
- CORS configuration for frontend access
- Docker Compose configuration validation
