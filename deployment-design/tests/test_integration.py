#!/usr/bin/env python3
"""
Integration tests for deployment-design system.
Tests the complete workflow from indexing to search functionality.
"""

import pytest
import requests
import time
import json
import subprocess
import os
from typing import Dict, Any


class TestDeploymentIntegration:
    """Integration tests for the complete deployment system."""
    
    
    def test_elasticsearch_health(self, elasticsearch_url):
        """Test that Elasticsearch cluster is healthy."""
        try:
            response = requests.get(f"{elasticsearch_url}/_cluster/health", timeout=10)
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] in ["green", "yellow"]
            assert health_data["number_of_nodes"] >= 1
            
        except requests.exceptions.RequestException:
            pytest.skip("Elasticsearch not available for integration testing")
    
    def test_elasticsearch_nodes(self, elasticsearch_url):
        """Test that Elasticsearch nodes are running."""
        try:
            response = requests.get(f"{elasticsearch_url}/_cat/nodes?format=json", timeout=10)
            assert response.status_code == 200
            
            nodes = response.json()
            assert len(nodes) >= 1
            
            # Check that at least one node is master
            master_nodes = [node for node in nodes if 'm' in node.get('node.role', '')]
            assert len(master_nodes) >= 1
            
        except requests.exceptions.RequestException:
            pytest.skip("Elasticsearch not available for integration testing")
    
    def test_index_creation_and_data_insertion(self, elasticsearch_url, test_index_name):
        """Test creating an index and inserting test data."""
        try:
            # Delete test index if it exists
            requests.delete(f"{elasticsearch_url}/{test_index_name}")
            
            # Create test index with mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "generated_text": {"type": "text", "analyzer": "standard"},
                        "age": {"type": "keyword"},
                        "gender": {"type": "keyword"},
                        "accent": {"type": "keyword"},
                        "duration": {"type": "float"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
            
            response = requests.put(
                f"{elasticsearch_url}/{test_index_name}",
                json=mapping,
                timeout=10
            )
            assert response.status_code in [200, 201]
            
            # Insert test documents
            test_docs = [
                {
                    "generated_text": "Hello world this is a test",
                    "age": "twenties",
                    "gender": "female",
                    "accent": "us",
                    "duration": 5.5
                },
                {
                    "generated_text": "Goodbye cruel world",
                    "age": "thirties",
                    "gender": "male",
                    "accent": "england",
                    "duration": 3.2
                },
                {
                    "generated_text": "How are you today",
                    "age": "twenties",
                    "gender": "female",
                    "accent": "us",
                    "duration": 4.1
                }
            ]
            
            for i, doc in enumerate(test_docs):
                response = requests.post(
                    f"{elasticsearch_url}/{test_index_name}/_doc/{i}",
                    json=doc,
                    timeout=10
                )
                assert response.status_code in [200, 201]
            
            # Refresh index to make documents searchable
            requests.post(f"{elasticsearch_url}/{test_index_name}/_refresh", timeout=10)
            
            # Verify document count
            response = requests.get(f"{elasticsearch_url}/{test_index_name}/_count", timeout=10)
            assert response.status_code == 200
            assert response.json()["count"] == 3
            
        except requests.exceptions.RequestException:
            pytest.skip("Elasticsearch not available for integration testing")
    
    def test_search_functionality(self, elasticsearch_url, test_index_name):
        """Test search functionality on indexed data."""
        try:
            # Test basic search
            search_query = {
                "query": {
                    "match": {
                        "generated_text": "world"
                    }
                }
            }
            
            response = requests.post(
                f"{elasticsearch_url}/{test_index_name}/_search",
                json=search_query,
                timeout=10
            )
            assert response.status_code == 200
            
            search_results = response.json()
            assert search_results["hits"]["total"]["value"] >= 1
            
            # Test faceted search
            facet_query = {
                "query": {"match_all": {}},
                "aggs": {
                    "ages": {
                        "terms": {"field": "age"}
                    },
                    "genders": {
                        "terms": {"field": "gender"}
                    }
                }
            }
            
            response = requests.post(
                f"{elasticsearch_url}/{test_index_name}/_search",
                json=facet_query,
                timeout=10
            )
            assert response.status_code == 200
            
            facet_results = response.json()
            assert "aggregations" in facet_results
            assert "ages" in facet_results["aggregations"]
            assert "genders" in facet_results["aggregations"]
            
        except requests.exceptions.RequestException:
            pytest.skip("Elasticsearch not available for integration testing")
    
    def test_range_search_on_duration(self, elasticsearch_url, test_index_name):
        """Test range search on duration field."""
        try:
            range_query = {
                "query": {
                    "range": {
                        "duration": {
                            "gte": 3.0,
                            "lte": 5.0
                        }
                    }
                }
            }
            
            response = requests.post(
                f"{elasticsearch_url}/{test_index_name}/_search",
                json=range_query,
                timeout=10
            )
            assert response.status_code == 200
            
            search_results = response.json()
            assert search_results["hits"]["total"]["value"] >= 1
            
            # Verify all returned documents are within the range
            for hit in search_results["hits"]["hits"]:
                duration = hit["_source"]["duration"]
                assert 3.0 <= duration <= 5.0
                
        except requests.exceptions.RequestException:
            pytest.skip("Elasticsearch not available for integration testing")
    
    def test_search_ui_availability(self, search_ui_url):
        """Test that search UI is accessible."""
        try:
            response = requests.get(search_ui_url, timeout=10)
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")
            
        except requests.exceptions.RequestException:
            pytest.skip("Search UI not available for integration testing")
    
    def test_cors_configuration(self, elasticsearch_url):
        """Test CORS configuration for search UI access."""
        try:
            headers = {
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
            
            response = requests.options(
                f"{elasticsearch_url}/_search",
                headers=headers,
                timeout=10
            )
            
            # Should not be blocked by CORS
            assert response.status_code in [200, 204]
            
        except requests.exceptions.RequestException:
            pytest.skip("Elasticsearch not available for integration testing")
    
    def test_cleanup_test_data(self, elasticsearch_url, test_index_name):
        """Clean up test data after tests."""
        try:
            # Delete test index
            response = requests.delete(f"{elasticsearch_url}/{test_index_name}", timeout=10)
            # 404 is acceptable if index doesn't exist
            assert response.status_code in [200, 404]
            
        except requests.exceptions.RequestException:
            # Cleanup failure should not fail the test
            pass


class TestDockerComposeIntegration:
    """Test docker-compose setup and service interactions."""
    
    def test_docker_compose_file_validity(self):
        """Test that docker-compose.yml is valid."""
        try:
            # Get the parent directory (deployment-design) where docker-compose.yml is located
            test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            result = subprocess.run(
                ["docker", "compose", "config"],
                cwd=test_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker Compose not available")
    
    def test_service_dependencies(self):
        """Test that service dependencies are correctly configured."""
        # Get the parent directory (deployment-design) where docker-compose.yml is located
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compose_file = os.path.join(test_dir, "docker-compose.yml")
        
        if not os.path.exists(compose_file):
            pytest.skip("Docker compose file not found")
        
        with open(compose_file, 'r') as f:
            content = f.read()
            
        # Verify key service dependencies
        assert "elasticsearch-node1" in content
        assert "elasticsearch-node2" in content
        assert "elastic-backend" in content
        assert "search-ui" in content
        
        # Verify dependency relationships
        assert "depends_on:" in content
        assert "condition: service_healthy" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])