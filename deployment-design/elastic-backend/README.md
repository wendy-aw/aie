# Elasticsearch Backend for ASR Search

This directory contains a 2-node Elasticsearch cluster setup for indexing and searching ASR (Automatic Speech Recognition) data from the CV dataset.

## Setup

### 1. Start Elasticsearch Cluster

Start the 2-node Elasticsearch cluster accessible on port 9200:

```bash
docker compose up -d
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Index the ASR Data

Run the indexing script to load the CSV data into Elasticsearch:

```bash
python cv-index.py
```

By default, this will:
- Index data from `../../asr/cv-valid-dev-updated.csv`
- Create an index named `cv-transcriptions`
- Connect to Elasticsearch at `localhost:9200`

#### Custom Options

```bash
# Custom CSV file and index name
python cv-index.py --csv-file /path/to/your/data.csv --index-name my-asr-index

# Custom Elasticsearch connection
python cv-index.py --host elasticsearch-host --port 9200

# Custom batch size for indexing
python cv-index.py --batch-size 500
```

## Data Schema

The indexed documents contain the following fields:

- `generated_text` (text): ASR-generated transcription  
- `age` (keyword): Speaker age group
- `gender` (keyword): Speaker gender
- `accent` (keyword): Speaker accent/region
- `duration` (float): Audio duration in seconds

## Usage Examples

### Search via Elasticsearch API

```bash
# Search for documents containing "careful"
curl -X GET "localhost:9200/cv-transcriptions/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match": {
      "generated_text": "careful"
    }
  }
}'

# Filter by speaker demographics
curl -X GET "localhost:9200/cv-transcriptions/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {"match": {"generated_text": "stranger"}},
        {"term": {"gender": "female"}},
        {"term": {"accent": "england"}}
      ]
    }
  }
}'
```


## Cluster Management

### Check Cluster Health

```bash
curl -X GET "localhost:9200/_cluster/health?pretty"
```

### View Index Statistics

```bash
curl -X GET "localhost:9200/cv-transcriptions/_stats?pretty"
```

### Shutdown

```bash
docker-compose down
```

To remove all data volumes:

```bash
docker-compose down -v
```

## Configuration

The Elasticsearch cluster is configured with:
- 2 nodes for high availability
- Security disabled for development
- 1GB heap size per node
- Index with 2 shards and 1 replica
- Custom mapping optimized for ASR data search