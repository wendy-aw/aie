#!/bin/bash

# Set default values
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8001}
WORKERS=${WORKERS:-$((2 * $(nproc) + 1))}

echo "Starting ASR API server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"

# Run uvicorn with workers
uvicorn asr_api:app --host $HOST --port $PORT --log-level warning --access-log --loop uvloop --http httptools --workers $WORKERS

echo "ASR API server stopped."