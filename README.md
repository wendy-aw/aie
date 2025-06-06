# AI-Powered Speech Recognition and Search Platform

This platform converts audio files into searchable text transcriptions and provides a web interface to search through them. It consists of two main components:

1. **ASR API**: Converts MP3 audio files to text with Facebook's Wav2Vec2 model.
2. **Search Platform**: Indexes transcriptions in Elasticsearch and provides a React web UI for full-text search with demographic filtering.

**Use Cases**: Search through podcast archives, meeting recordings, interview databases, or any large collection of audio content by text content and characteristics.

The search platform is deployed at http://52.87.214.55:3000/.

## Repository Structure

```
aie/
├── asr/                    # ASR API Service
├── deployment-design/      # Complete Search Platform
│   ├── elastic-backend/    # Data indexing service
│   ├── search-ui/          # React web interface

```

## Components

### ASR (Automatic Speech Recognition)
**Location**: `/asr/`  
**Purpose**: API service for speech-to-text conversion

**Key Features**:
- FastAPI-based API for audio transcription
- Docker containerization with health checks

**Main Files**:
- `asr_api.py` - Main FastAPI application
- `cv-decode.py` - Core ASR processing logic
- `startup.sh` - Script to start the ASR API
- `docker-compose.yml` - Docker deployment configuration

**Usage**:
```bash
cd asr
docker compose up --build
# API available at http://localhost:8001
```
For more details, see the [ASR API documentation](asr/README.md).

### Deployment Design (Search Platform)
**Location**: `/deployment-design/`  
**Purpose**: Complete search platform with Elasticsearch backend and React frontend

**Key Features**:
- Full-text search across speech transcriptions
- Faceted filtering by speaker demographics (age, gender, accent, duration)
- Scalable indexing with bulk data processing
- Multi-container orchestration with 2-node Elasticsearch cluster for high availability and scalability

**Main Components**:
- `elastic-backend/` - Data indexing service and Elasticsearch configuration
- `search-ui/` - React frontend with Elastic Search UI components
- `docker-compose.yml` - Complete stack orchestration

**Usage**:
```bash
cd deployment-design
docker compose up --build
# Search UI: http://localhost:3000
# Elasticsearch: http://localhost:9200
```
For more details, see the [Search Platform documentation](deployment-design/README.md).

## Data Flow

1. **Audio Processing**: ASR service processes audio files → generates transcriptions with model and saves to CSV
2. **Data Indexing**: Elastic backend ingests CSV transcription data → indexes into Elasticsearch
3. **Search & Discovery**: Users interact with Search UI → queries Elasticsearch cluster → displays results

