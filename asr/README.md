# Task 2: ASR (Automatic Speech Recognition) Microservice

A high-performance FastAPI-based microservice for transcribing MP3 audio files using Facebook's wav2vec 2.0 model, with batch processing capabilities and a script for processing the Common Voice dataset.

## Features

- **FastAPI REST API** with single file and batch transcription endpoints
- **Batch processing** with true tensor batching for improved performance
- **Scalable architecture** with configurable worker processes
- **GPU/MPS support** with automatic device detection
- **Docker containerization** for easy deployment
- **Client script** for processing Common Voice CSV datasets
- **Comprehensive error handling** and logging
- **Environment-based configuration**

## Quick Start

### 1. Installation

```bash
# Install dependencies
cd asr
pip install -r requirements.txt
```

### 2. Configuration

Customize the environment file `asr/.env`.

### 3. Start the API Server

#### Option 1: Local Deployment

**Prerequisites:**
- Python 3.8+
- FFmpeg (required for audio processing)

Install FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

Start the server:
```bash
# Make startup script executable
chmod +x startup.sh

# Start server
./startup.sh
```

The startup script will:
- Load environment variables from [.env](./.env) file
- Auto-detect CPU cores and set optimal worker count if `WORKERS` is not set in [.env](./.env)
- Start the server with proper configuration
- Launch the ASR API server (`asr_api.py`) using Uvicorn, serving the API on the configured host and port

#### Option 2: Docker Deployment

> **Prerequisite:**  
> Make sure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.

```bash
# Build and run with docker-compose
docker compose up --build
```

This will build the Docker image and start a container with the API server. The server will be exposed on port 8001 and can be accessed from the host machine.

### 4. Test the API

```bash
# Health check
curl http://localhost:8001/ping

# Single file transcription
curl -X POST "http://localhost:8001/asr" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/file.mp3"

# Batch transcription
curl -X POST "http://localhost:8001/asr" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@path/to/your/file1.mp3" \
  -F "files=@path/to/your/file2.mp3"
```

## API Endpoints

### GET /ping
Health check endpoint.

**Response:**
```json
{"message": "pong"}
```

### POST /asr
Transcribe audio files (single or batch).

**Parameters:**
- `file` (optional): Single MP3 file
- `files` (optional): Multiple MP3 files for batch processing

**Single File Response:**
```json
{
  "transcription": "hello world",
  "duration": "2.5"
}
```

**Batch Response:**
```json
{
  "results": [
    {
      "filename": "audio1.mp3",
      "status": "success",
      "transcription": "hello world",
      "duration": "2.5",
      "processing_time": 0.45
    },
    {
      "filename": "audio2.mp3",
      "status": "error",
      "transcription": "",
      "error": "File too large",
      "processing_time": 0.01
    }
  ]
}
```

## Transcription Script (cv-decode.py)

Process the Common Voice dataset efficiently using ASR API calls.

The script uses batching and concurrency for optimal performance:
- Audio files are grouped and sent in batches, reducing the number of HTTP requests and improving throughput.
- By batching audio files, the ASR model can also process multiple files simultaneously, reducing the total inference time.
- Multiple batches are processed concurrently, leveraging asynchronous requests to maximize speed and server utilization.


### Configuration
Set your configurations in the `.env` file and change the `DATA_FOLDER` path to the folder containing your MP3 files

### Usage

```bash
# Basic usage with default parameters
python cv-decode.py

# Custom parameters
python cv-decode.py \
  --csv cv-valid-dev.csv \
  --folder cv-valid-dev \
  --output ../deployment-design/elastic-backend/csv_to_index.csv \
  --concurrent 10 \
  --batch_size 8 \
  --n_files 100
```

### CLI Parameters

- `--csv`: Input CSV file with audio filenames (default: cv-valid-dev.csv)
- `--folder`: Folder containing MP3 files (default: cv-valid-dev)
- `--output`: Output CSV file with transcriptions (default: ../deployment-design/elastic-backend/csv_to_index.csv)
- `--concurrent`: Max concurrent requests (default: 2x API workers)
- `--batch_size`: Number of files per batch (default: 5)
- `--n_files`: Limit number of files to process

**Note:** Long audio samples may cause batch processing to fail due to timeout or memory constraints. Consider using smaller batch sizes (`--batch_size 1-3`) when processing files longer than 20 seconds.

All default parameters can be managed through the [`.env`](./.env) file:


## Monitoring and Logging

### Log Files

- **API logs**: `logs/asr_api.log`
- **Client logs**: `logs/cv-decode.log`

### Log Levels

Set via `LOG_LEVEL` environment variable:
- `DEBUG`: Detailed processing information
- `INFO`: General operational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages only

## Testing

The project includes comprehensive tests covering unit tests, integration tests, and performance tests.

### Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m slow             # Performance/slow tests
pytest -m "not slow"       # Skip slow tests

# Run with coverage report
pytest --cov=asr_api --cov=cv-decode --cov-report=html

# Run specific test files
pytest tests/unit/test_audio_processing.py
pytest tests/integration/test_api_endpoints.py
```

### Test Structure

```
tests/
├── conftest.py                     # Test configuration and fixtures
├── fixtures/                      # Test data files
│   └── test_data.csv
├── unit/                          # Unit tests
│   ├── test_audio_processing.py   # Audio processing functions
│   └── test_cv_decode.py          # CV-decode functionality
└── integration/                   # Integration tests
    ├── test_api_endpoints.py      # API endpoint tests
    ├── test_performance.py        # Performance tests
    └── test_end_to_end.py         # End-to-end workflows
```


### Integration Test Requirements

Some integration tests require:

- **API Server Running**: For end-to-end tests
- **Docker**: For container workflow tests
- **Network Access**: For real API calls

Skip these tests if dependencies are unavailable:
```bash
pytest -m "not integration"
```

