# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies, ffmpeg for mp3 processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY asr_api.py .
COPY startup.sh .

# Create logs directory
RUN mkdir -p logs

# Make startup script executable
RUN chmod +x startup.sh

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash asr-user && \
    chown -R asr-user:asr-user /app

# Switch to non-root user
USER asr-user

# Expose port (default to 8001 if PORT not set)
EXPOSE ${PORT:-8001}

# Use startup script as entrypoint
CMD ["./startup.sh"]