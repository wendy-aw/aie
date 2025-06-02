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
pip install -r requirements.txt
```

### 2. Configuration

Customize the environment file `.env`

### 3. Prepare audio files

Add your audio files into the `cv-valid-dev` folder or change the `DATA_FOLDER` path in the `.env` file.

### 4. Start the API Server

#### Option 1: Local Deployment

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
docker-compose up --build
```

This will build the Docker image and start a container with the API server. The server will be exposed on port 8001 and can be accessed from the host machine.

### 5. Test the API

```bash
# Health check
curl http://localhost:8001/ping

# Single file transcription
curl -X POST "http://localhost:8001/asr" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@asr/cv-valid-dev/sample-000000.mp3"

# Batch transcription
curl -X POST "http://localhost:8001/asr" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@asr/cv-valid-dev/sample-000000.mp3" \
  -F "files=@asr/cv-valid-dev/sample-000001.mp3"
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

### Usage

```bash
# Basic usage with default parameters
python cv-decode.py

# Custom parameters
python cv-decode.py \
  --csv cv-valid-dev.csv \
  --folder cv-valid-dev \
  --output cv-valid-dev-updated.csv \
  --concurrent 10 \
  --batch_size 8 \
  --n_files 100
```

### CLI Parameters

- `--csv`: Input CSV file with audio filenames (default: cv-valid-dev.csv)
- `--folder`: Folder containing MP3 files (default: cv-valid-dev)
- `--output`: Output CSV file with transcriptions (default: cv-valid-dev-updated.csv)
- `--concurrent`: Max concurrent requests (default: 2x API workers)
- `--batch_size`: Number of files per batch (default: 5)
- `--n_files`: Limit number of files to process

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
