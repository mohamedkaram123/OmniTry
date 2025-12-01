# Dockerfile for OmniTry on RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements_api.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_api.txt

# Optional: Install flash-attention for better performance (requires compilation)
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation || echo "Flash attention install failed, continuing without it"

# Copy application code
COPY . /app/

# Create checkpoints directory
RUN mkdir -p /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_TEMP_DIR=/tmp/.gradio

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for RunPod serverless
CMD ["python", "-u", "runpod_handler.py"]
