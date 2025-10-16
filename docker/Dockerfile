# MABE Pipeline Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p outputs/models outputs/submissions outputs/logs outputs/studies

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MABE_DEVICE__USE_CUDA=false
ENV MABE_DEVICE__DEVICE_STR=cpu

# Make scripts executable
RUN chmod +x bin/run_pipeline.py
RUN chmod +x scripts/ci_run_short.sh
RUN chmod +x scripts/examples/run_example.sh

# Default command
CMD ["python", "bin/run_pipeline.py", "--help"]
