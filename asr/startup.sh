#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Set fallback defaults (in case .env is missing)
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8001}

# If workers is not set in .env, get number of CPU cores and set workers to 2 * CPU_CORES + 1
if command -v nproc >/dev/null 2>&1; then
  CPU_CORES=$(nproc)
else
  CPU_CORES=$(sysctl -n hw.ncpu)
fi
WORKERS=${WORKERS:-$((2 * CPU_CORES + 1))}

echo "Starting ASR API server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"

# Run uvicorn with workers
uvicorn asr_api:app --host $HOST --port $PORT --log-level warning --access-log --loop uvloop --http httptools --workers $WORKERS

echo "ASR API server stopped."