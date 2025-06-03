# FROM python:3.10
#
# WORKDIR /app
# COPY . .
#
# RUN pip install -r requirements.txt
#
# CMD ["python", "-u", "server.py"]


# Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download model (or mount via volume)
RUN mkdir -p /app/models && \
    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -O ${MODEL_PATH}

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "server.py"]