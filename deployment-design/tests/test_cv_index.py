#!/usr/bin/env python3
"""
Unit tests for cv_index.py elastic backend functionality.
"""

import pytest
import os
from unittest.mock import Mock, patch

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elastic-backend'))
from cv_index import (
    create_elasticsearch_client,
    create_index_mapping,
    create_index,
    process_csv_row,
    generate_docs,
    index_csv_data,
    verify_indexing
)


class TestElasticsearchClient:
    """Test Elasticsearch client creation and connection."""
    
    @patch('cv_index.Elasticsearch')
    def test_create_elasticsearch_client_success(self, mock_es_class):
        """Test successful connection to Elasticsearch."""
        mock_es = Mock()
        mock_es.ping.return_value = True
        mock_es_class.return_value = mock_es
        
        client = create_elasticsearch_client("localhost", 9200)
        
        assert client == mock_es
        mock_es_class.assert_called_once_with(["http://localhost:9200"])
        mock_es.ping.assert_called_once()
    
    @patch('cv_index.Elasticsearch')
    @patch('cv_index.time.sleep')
    def test_create_elasticsearch_client_failure(self, mock_sleep, mock_es_class):
        """Test connection failure after retries."""
        mock_es = Mock()
        mock_es.ping.side_effect = Exception("Connection failed")
        mock_es_class.return_value = mock_es
        
        with pytest.raises(ConnectionError):
            create_elasticsearch_client("localhost", 9200)
        
        assert mock_sleep.call_count == 60


class TestIndexMapping:
    """Test index mapping creation."""
    
    def test_create_index_mapping(self):
        """Test that index mapping is correctly structured."""
        mapping = create_index_mapping()
        
        assert "mappings" in mapping
        assert "settings" in mapping
        
        properties = mapping["mappings"]["properties"]
        assert properties["age"]["type"] == "keyword"
        assert properties["gender"]["type"] == "keyword"
        assert properties["accent"]["type"] == "keyword"
        assert properties["duration"]["type"] == "float"
        assert properties["generated_text"]["type"] == "text"
        assert properties["generated_text"]["analyzer"] == "standard"
        
        settings = mapping["settings"]
        assert settings["number_of_shards"] == 2
        assert settings["number_of_replicas"] == 1


class TestIndexCreation:
    """Test index creation functionality."""
    
    def test_create_index_new(self):
        """Test creating a new index."""
        mock_es = Mock()
        mock_es.indices.exists.return_value = False
        
        create_index(mock_es, "test-index")
        
        mock_es.indices.exists.assert_called_once_with(index="test-index")
        mock_es.indices.create.assert_called_once()
        _, kwargs = mock_es.indices.create.call_args
        assert kwargs["index"] == "test-index"
        assert "body" in kwargs
    
    def test_create_index_existing(self):
        """Test creating index when it already exists."""
        mock_es = Mock()
        mock_es.indices.exists.return_value = True
        
        create_index(mock_es, "test-index")
        
        mock_es.indices.delete.assert_called_once_with(index="test-index")
        mock_es.indices.create.assert_called_once()


class TestCSVProcessing:
    """Test CSV data processing functionality."""
    
    def test_process_csv_row_complete(self):
        """Test processing a complete CSV row."""
        row = {
            'generated_text': 'Hello world',
            'age': 'twenties',
            'gender': 'female',
            'accent': 'us',
            'duration': '5.5'
        }
        
        doc = process_csv_row(row)
        
        assert doc['generated_text'] == 'Hello world'
        assert doc['age'] == 'twenties'
        assert doc['gender'] == 'female'
        assert doc['accent'] == 'us'
        assert doc['duration'] == 5.5
    
    def test_process_csv_row_empty_fields(self):
        """Test processing CSV row with empty fields."""
        row = {
            'generated_text': 'Hello world',
            'age': '',
            'gender': '',
            'accent': '',
            'duration': ''
        }
        
        doc = process_csv_row(row)
        
        assert doc['generated_text'] == 'Hello world'
        assert doc['age'] is None
        assert doc['gender'] is None
        assert doc['accent'] is None
        assert doc['duration'] is None
    
    def test_process_csv_row_invalid_duration(self):
        """Test processing CSV row with invalid duration."""
        row = {
            'generated_text': 'Hello world',
            'age': 'twenties',
            'gender': 'female',
            'accent': 'us',
            'duration': 'invalid'
        }
        
        doc = process_csv_row(row)
        
        assert doc['duration'] is None


class TestDocumentGeneration:
    """Test document generation for indexing."""
    
    def test_generate_docs(self, temp_csv_file):
        """Test document generation from CSV file."""
        docs = list(generate_docs(temp_csv_file, "test-index"))
        
        assert len(docs) == 3
        
        # Check first document
        doc1 = docs[0]
        assert doc1['_index'] == 'test-index'
        assert doc1['_source']['generated_text'] == 'Hello world test'
        assert doc1['_source']['age'] == 'twenties'
        assert doc1['_source']['duration'] == 5.5
        
        # Check second document
        doc2 = docs[1]
        assert doc2['_index'] == 'test-index'
        assert doc2['_source']['generated_text'] == 'Goodbye test'
        assert doc2['_source']['age'] == 'thirties'
        assert doc2['_source']['duration'] == 3.2


class TestIndexingOperations:
    """Test data indexing operations."""
    
    @patch('cv_index.bulk')
    @patch('cv_index.tqdm')
    def test_index_csv_data_success(self, mock_tqdm, mock_bulk, temp_csv_file):
        """Test successful CSV data indexing."""
        mock_es = Mock()
        mock_es.options.return_value = mock_es
        mock_bulk.return_value = (1, [])
        
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        
        index_csv_data(mock_es, temp_csv_file, "test-index", 1000)
        
        mock_bulk.assert_called_once()
        mock_progress.close.assert_called()
    
    @patch('cv_index.bulk')
    @patch('cv_index.tqdm')
    def test_index_csv_data_with_failures(self, mock_tqdm, mock_bulk, temp_csv_file):
        """Test CSV data indexing with some failures."""
        mock_es = Mock()
        mock_es.options.return_value = mock_es
        mock_bulk.return_value = (0, [{'error': 'test error'}])
        
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        
        index_csv_data(mock_es, temp_csv_file, "test-index", 1000)
        
        mock_bulk.assert_called_once()
        mock_progress.close.assert_called()


class TestIndexVerification:
    """Test index verification functionality."""
    
    def test_verify_indexing(self):
        """Test index verification after indexing."""
        mock_es = Mock()
        mock_es.count.return_value = {'count': 100}
        mock_es.search.return_value = {
            'hits': {
                'total': {'value': 5},
                'hits': [{
                    '_source': {
                        'generated_text': 'test text',
                        'age': 'twenties'
                    }
                }]
            }
        }
        
        verify_indexing(mock_es, "test-index")
        
        mock_es.indices.refresh.assert_called_once_with(index="test-index")
        mock_es.count.assert_called_once_with(index="test-index")
        mock_es.search.assert_called_once()