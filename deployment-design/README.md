# Task 3: Elastic Backend & Search UI

This directory contains the search platform components for indexing and searching transcription data on a user interface.

## Components

### Elastic Backend
**Purpose**: Automatically indexes CSV transcription data into Elasticsearch

The indexer can read CSV data from:
- **Remote URLs** (HTTP/HTTPS, S3, Google Cloud Storage) - Recommended for cloud deployments
- **Local files** - For local deployments, copy your CSV data into the `elastic-backend/csv_to_index.csv` file. This file will be read by the indexer when the container starts.

**CSV Format**: The data should have these columns:
- `generated_text` - Transcribed speech content (text)
- `age` - Speaker age category (keyword: "twenties", "thirties", etc.)  
- `gender` - Speaker gender (keyword: "male", "female", etc.)
- `accent` - Speaker accent/region (keyword: "us", "england", etc.)
- `duration` - Audio duration in seconds (float: 2.5, 10.3, etc.)

### Search UI
**Purpose**: React web interface for searching and filtering transcriptions

![Search UI Screenshot](search_ui.png)

## Setup & Usage

### Quick Start

1. **Configure CSV Source**: For remote CSV sources, edit `.env` file and set your CSV URL:
   ```bash
   # Example URLs:
   CSV_SOURCE_URL=https://example.com/data.csv
   CSV_SOURCE_URL=s3://bucket-name/transcriptions.csv
   CSV_SOURCE_URL=https://raw.githubusercontent.com/user/repo/main/data.csv
   ```
For local CSV files, leave `CSV_SOURCE_URL` empty. You can use the provided `elastic-backend/csv_to_index.csv` file or replace it with your own file. This will be mounted at `/app/csv_to_index.csv` inside the container.

2. **Start Platform**: This starts all services and indexes your data:
   ```bash
   docker compose up --build
   ```

3. **Access Services**:
   - Search UI: http://localhost:3000
   - Elasticsearch: http://localhost:9200

The platform will automatically:
- Start Elasticsearch cluster
- Download and index CSV data from your URL or local file
- Start Search UI

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
For development or custom indexing scenarios:

```bash
cd elastic-backend
pip install -r requirements.txt

# Index from URL
export CSV_SOURCE_URL=https://example.com/data.csv
python cv_index.py --host localhost --port 9200

# Or specify directly
python cv_index.py --csv-file https://example.com/data.csv --host localhost
```

**Options**:
- `--csv-file`: CSV file path or URL (overrides CSV_SOURCE_URL env var)
- `--host`: Elasticsearch host (default: localhost)
- `--port`: Elasticsearch port (default: 9200)
- `--index-name`: Index name (default: cv-transcriptions)
- `--batch-size`: Batch size for indexing (default: 1000)

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