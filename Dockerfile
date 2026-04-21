# ── Stage 1: Base ────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data models

# ── NOTE ─────────────────────────────────────────────────────────────────────
# The model must be trained BEFORE building this image, OR
# you can run training inside the container after mounting your data:
#
#   docker exec -it loan-api python src/train.py
#
# OR mount a pre-trained models/ directory:
#   docker run -v $(pwd)/models:/app/models ...
# ─────────────────────────────────────────────────────────────────────────────

# Expose port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
