# Task 3: Elastic Backend & Search UI

This directory contains the search platform components for indexing and searching transcription data on a user interface.

## Components

### Elastic Backend
**Purpose**: Automatically indexes CSV transcription data into Elasticsearch

The indexer expects a CSV file named `csv_to_index.csv` with the following columns:
- `generated_text` - Transcribed speech content (text)
- `age` - Speaker age category (keyword: "twenties", "thirties", etc.)  
- `gender` - Speaker gender (keyword: "male", "female", etc.)
- `accent` - Speaker accent/region (keyword: "us", "england", etc.)
- `duration` - Audio duration in seconds (float: 2.5, 10.3, etc.)

Use the provided `elastic-backend/csv_to_index.csv` file or replace it with your own data following the same format.

### Search UI
**Purpose**: React web interface for searching and filtering transcriptions

![Search UI Screenshot](search_ui.png)

## Setup & Usage

### Quick Start

This starts the complete platform with all services. It will do the following:
- Start Elasticsearch cluster
- Index CSV data into Elasticsearch
- Start Search UI

```bash
# Start complete platform
docker compose up --build

# Access services
# Search UI: http://localhost:3000
# Elasticsearch: http://localhost:9200
```

### Development Mode
You can start individual services if you want to manage them separately.

```bash
# Elasticsearch cluster only
docker compose up elasticsearch-node1 elasticsearch-node2

# Indexing only (requires Elasticsearch running)
docker compose up elastic-backend

# Search UI only (requires data indexed)
docker compose up search-ui
```

### Manual Indexing
In case you want to index the data manually (without using docker compose), you can run the following command:
```bash
cd elastic-backend
pip install -r requirements.txt
# Ensure csv_to_index.csv is present before running cv_index.py
python cv_index.py --host localhost --port 9200
```

**Custom Options**:
- `--host`: Elasticsearch host (default: es-node1 for Docker, localhost for local)
- `--port`: Elasticsearch port (default: 9200)
- `--index`: Index name (default: cv-transcriptions)
- `--batch-size`: Batch size for indexing (default: 1000)
- `--csv-file`: Path to CSV file for indexing (default: csv_to_index.csv)

### Search UI Development
Before running the search UI, make sure that the Elasticsearch cluster is running and the data is indexed.
```bash
cd search-ui
npm install
npm start
# Available at http://localhost:3000 (development server)
```

## Index Management

### Useful Commands
```bash
# Check cluster health
curl -X GET "http://localhost:9200/_cluster/health?pretty"

# View index stats
curl -X GET "http://localhost:9200/cv-transcriptions/_stats?pretty"

# Count documents
curl -X GET "http://localhost:9200/cv-transcriptions/_count"

# Delete and recreate index
curl -X DELETE "http://localhost:9200/cv-transcriptions"
docker compose up elastic-backend
```

### Data Updates
To update indexed data:
1. Stop services: `docker compose down`
2. Replace `csv_to_index.csv` with new data
3. Restart: `docker compose up --build`

The elastic-backend will automatically delete existing index and recreate with new data.