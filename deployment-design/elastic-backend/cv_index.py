#!/usr/bin/env python3
"""
Script to index ASR CSV data into Elasticsearch cluster.
"""

import csv
import json
import os
import tempfile
from typing import Dict, Any
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import argparse
import logging
import time
import requests
from urllib.parse import urlparse


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_elasticsearch_client(host: str = "localhost", port: int = 9200) -> Elasticsearch:
    """Create and return Elasticsearch client."""
    
    es = Elasticsearch([f"http://{host}:{port}"])
    
    # Wait for Elasticsearch to be ready with retries
    max_retries = 60
    for attempt in range(max_retries):
        try:
            if es.ping():
                logging.info(f"Connected to Elasticsearch at {host}:{port}")
                return es
        except Exception:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Waiting for Elasticsearch to be ready...")
            time.sleep(3)
    
    raise ConnectionError(f"Could not connect to Elasticsearch at {host}:{port} after {max_retries} attempts")


def create_index_mapping() -> Dict[str, Any]:
    """Define the mapping for the ASR index."""
    return {
        "mappings": {
            "properties": {
                "age": {
                    "type": "keyword"
                },
                "gender": {
                    "type": "keyword"
                },
                "accent": {
                    "type": "keyword"
                },
                "duration": {
                    "type": "float"
                },
                "generated_text": {
                    "type": "text",
                    "analyzer": "standard"
                }
            }
        },
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1
        }
    }


def create_index(es: Elasticsearch, index_name: str) -> None:
    """Create the index with proper mapping."""
    if es.indices.exists(index=index_name):
        logging.info(f"Index {index_name} already exists. Deleting...")
        es.indices.delete(index=index_name)
    
    mapping = create_index_mapping()
    es.indices.create(index=index_name, body=mapping)
    logging.info(f"Created index {index_name}")


def process_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Process a single CSV row into Elasticsearch document."""
    doc = {}
    
    # Handle all fields
    doc['generated_text'] = row['generated_text']
    doc['age'] = row['age'] if row['age'] else None
    doc['gender'] = row['gender'] if row['gender'] else None
    doc['accent'] = row['accent'] if row['accent'] else None
    
    try:
        doc['duration'] = float(row['duration']) if row['duration'] else None
    except ValueError:
        doc['duration'] = None
    
    return doc


def download_csv_from_url(url: str) -> str:
    """Download CSV file from URL to temporary file."""
    logging.info(f"Downloading CSV from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.csv')
        
        # Download with progress bar
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
            progress_bar.update(len(chunk))
        
        progress_bar.close()
        temp_file.close()
        
        logging.info(f"Downloaded CSV to temporary file: {temp_file.name}")
        return temp_file.name
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download CSV from {url}: {e}")
        raise


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ('http', 'https', 's3', 'gs')


def generate_docs(csv_file_path: str, index_name: str):
    """Generator function to yield documents for bulk indexing."""
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            doc = process_csv_row(row)
            yield {
                '_index': index_name,
                '_source': doc
            }


def index_csv_data(es: Elasticsearch, csv_source: str, index_name: str, batch_size: int = 1000) -> None:
    """Index CSV data into Elasticsearch using bulk operations."""
    logging.info(f"Starting to index data from {csv_source}")
    
    # Download file if it's a URL (only once)
    if is_url(csv_source):
        csv_file_path = download_csv_from_url(csv_source)
        cleanup_temp = True
    else:
        csv_file_path = csv_source
        cleanup_temp = False
    
    try:
        # Count total rows for progress tracking
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            total_rows = sum(1 for _ in file) - 1  # Subtract header row
        
        logging.info(f"Total rows to index: {total_rows}")
        
        # Use bulk helper for efficient indexing
        progress_bar = tqdm(total=total_rows, desc="Indexing documents")
        
        def doc_generator():
            for doc in generate_docs(csv_file_path, index_name):
                progress_bar.update(1)
                yield doc
        
        # Bulk index with custom batch size
        es_with_options = es.options(request_timeout=60)
        success_count, failed_items = bulk(
            es_with_options,
            doc_generator(),
            chunk_size=batch_size,
        )
        
        progress_bar.close()
        logging.info(f"Successfully indexed {success_count} documents")
        
        if failed_items:
            logging.warning(f"Failed to index {len(failed_items)} documents")
            for item in failed_items[:5]:  # Show first 5 failures
                logging.warning(f"Failed item: {item}")
                
    except Exception as e:
        progress_bar.close()
        logging.error(f"Error during bulk indexing: {e}")
        raise
    finally:
        # Clean up temporary file if we downloaded one
        if cleanup_temp and os.path.exists(csv_file_path):
            os.unlink(csv_file_path)
            logging.info(f"Cleaned up temporary file: {csv_file_path}")


def verify_indexing(es: Elasticsearch, index_name: str) -> None:
    """Verify that data was indexed correctly."""
    # Refresh index to ensure all documents are searchable
    es.indices.refresh(index=index_name)
    
    # Get document count
    count_result = es.count(index=index_name)
    doc_count = count_result['count']
    
    logging.info(f"Total documents in index: {doc_count}")
    
    # Test search functionality
    test_search = es.search(
        index=index_name,
        body={
            "size": 3,
            "query": {
                "match": {
                    "generated_text": "careful"
                }
            }
        }
    )
    
    logging.info(f"Test search returned {test_search['hits']['total']['value']} results")

    if test_search['hits']['hits']:
        logging.info(f"Sample test document: {json.dumps(test_search['hits']['hits'][0]['_source'], indent=2)}")


def main():
    """Main function to orchestrate the indexing process."""
    parser = argparse.ArgumentParser(description='Index ASR CSV data into Elasticsearch')
    # Get CSV_SOURCE_URL and default to local file if empty
    csv_source_url = os.getenv('CSV_SOURCE_URL', '').strip()
    default_csv = csv_source_url if csv_source_url else 'csv_to_index.csv'
    
    parser.add_argument('--csv-file', default=default_csv,
                       help='Path to CSV file or URL (supports http/https/s3/gs) (default: CSV_SOURCE_URL env var or csv_to_index.csv)')
    parser.add_argument('--index-name', default='cv-transcriptions',
                       help='Elasticsearch index name (default: cv-transcriptions)')
    parser.add_argument('--host', default=os.getenv('ELASTICSEARCH_HOST', 'localhost'),
                       help='Elasticsearch host')
    parser.add_argument('--port', type=int, default=9200,
                       help='Elasticsearch port (default: 9200)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for bulk indexing (default: 1000)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # Create Elasticsearch client
        es = create_elasticsearch_client(args.host, args.port)
        
        # Create index with mapping
        create_index(es, args.index_name)
        
        # Index the data
        index_csv_data(es, args.csv_file, args.index_name, args.batch_size)
        
        # Verify indexing
        verify_indexing(es, args.index_name)

        # Close Elasticsearch client
        es.close()
        
        logging.info("Indexing completed successfully!")
        
        
    except Exception as e:
        logging.error(f"CRITICAL: Elastic backend failed - {e}")
        logging.error("WARNING: Search UI will not work without indexed data!")
        logging.error("To fix: Check Elasticsearch connection and restart this service")
        
        raise


if __name__ == "__main__":
    main()