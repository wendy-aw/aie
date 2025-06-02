#!/bin/bash

# ASR API Server Startup Script
# This script runs the ASR API with proper worker configuration

# Set default values
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8001}
WORKERS=${WORKERS:-$((2 * $(nproc) + 1))}
PROD=${PROD:-"false"}

echo "Starting ASR API server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"

# Build uvicorn command
UVICORN_CMD="uvicorn asr_api:app --host $HOST --port $PORT --log-level warning --access-log --loop uvloop --http httptools"

# Add reload for development
if [ "$PROD" = "false" ] || [ "$PROD" = "FALSE" ]; then
    echo "Development mode: enabling auto-reload"
    UVICORN_CMD="$UVICORN_CMD --reload"
else
    echo "Production mode: using $WORKERS workers"
    UVICORN_CMD="$UVICORN_CMD --workers $WORKERS"
fi

# Run uvicorn
eval $UVICORN_CMD

echo "ASR API server stopped."