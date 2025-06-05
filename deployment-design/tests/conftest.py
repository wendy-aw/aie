"""
Shared pytest fixtures for deployment-design tests.
"""

import pytest
import tempfile
import csv
import os


@pytest.fixture(scope="session")
def elasticsearch_url():
    """Elasticsearch URL for integration testing."""
    return "http://localhost:9200"


@pytest.fixture(scope="session") 
def search_ui_url():
    """Search UI URL for integration testing."""
    return "http://localhost:3000"


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return [
        ['generated_text', 'age', 'gender', 'accent', 'duration'],
        ['Hello world test', 'twenties', 'female', 'us', '5.5'],
        ['Goodbye test', 'thirties', 'male', 'england', '3.2'],
        ['How are you', 'twenties', 'female', 'us', '4.1']
    ]


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        writer = csv.writer(f)
        for row in sample_csv_data:
            writer.writerow(row)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture(scope="session")
def test_index_name():
    """Test index name for integration tests."""
    return "cv-transcriptions-test"