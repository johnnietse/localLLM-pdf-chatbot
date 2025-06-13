## FROM python:3.10
##
## WORKDIR /app
## COPY . .
##
## RUN pip install -r requirements.txt
##
## CMD ["python", "-u", "server.py"]
#
#
## Dockerfile
#FROM python:3.10-slim
#
## Set environment variables
#ENV MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#
## Install system dependencies
#RUN apt-get update && apt-get install -y \
#    poppler-utils \
#    libgl1 \
#    && rm -rf /var/lib/apt/lists/*
#
## Create app directory
#WORKDIR /app
#
## Copy requirements and install
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt
#
## Copy application files
#COPY . .
#
## Download model (or mount via volume)
#RUN mkdir -p /app/models && \
#    wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
#    -O ${MODEL_PATH}
#
## Expose port
#EXPOSE 8000
#
## Start application
#CMD ["python", "server.py"]



# # Use Python 3.9 as base image
# FROM python:3.9-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1
# ENV MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     poppler-utils \  # Required for pdf2image \
#     libgl1 \         # Required for pdf2image \
#     && rm -rf /var/lib/apt/lists/*

# # Create and set working directory
# WORKDIR /app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Create models directory
# RUN mkdir -p /app/models

# # Copy the model download script
# COPY download_models.ps1 .

# # Install PowerShell for model download (optional - remove if you'll manually provide models)
# RUN apt-get update && apt-get install -y wget \
#     && wget -q https://packages.microsoft.com/config/debian/10/packages-microsoft-prod.deb \
#     && dpkg -i packages-microsoft-prod.deb \
#     && apt-get update \
#     && apt-get install -y powershell \
#     && rm packages-microsoft-prod.deb

# # Download models (comment out if you'll mount models volume instead)
# # RUN pwsh ./download_models.ps1

# # Copy application files
# COPY . .

# # Expose the port the app runs on
# EXPOSE 8000

# # Command to run the application
# CMD ["python", "server.py"]



######################################################

# # Use Python 3.9 as base image
# FROM python:3.9-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     poppler-utils \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # Create and set working directory
# WORKDIR /app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Create models directory
# RUN mkdir -p /app/models



# RUN apt-get update && apt-get install -y wget \
#     && mkdir -p /app/models \
#     && wget -q https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O /app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
#     && apt-get remove -y wget \
#     && apt-get autoremove -y \
#     && rm -rf /var/lib/apt/lists/*


# # Copy application files
# COPY . .

# # Expose the port the app runs on
# EXPOSE 8000

# # Command to run the application
# CMD ["python", "server.py"]

################################################


# # Use Python 3.9 as base image
# FROM python:3.9-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     poppler-utils \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # Create and set working directory
# WORKDIR /app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Create models directory
# RUN mkdir -p /app/models

# # Optionally download the model automatically (uncomment if needed)
# # RUN apt-get update && apt-get install -y wget \
# #     && wget -q https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O ${MODEL_PATH} \
# #     && apt-get remove -y wget \
# #     && apt-get autoremove -y \
# #     && rm -rf /var/lib/apt/lists/*

# # Copy application files
# COPY . .

# # Expose the port the app runs on
# EXPOSE 8000

# # Command to run the application
# CMD ["python", "server.py"]

##############################################################################



# # Use Python 3.9 as base image
# FROM python:3.9-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     poppler-utils \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # Create and set working directory
# WORKDIR /app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# RUN mkdir -p /app/.cache /app/.chroma_db && \
#     chmod -R 777 /app/.cache /app/.chroma_db


    

# # Create models directory
# RUN mkdir -p /app/models

# # Download the model automatically (recommended)
# RUN apt-get update && apt-get install -y wget \
#     && wget -q https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -O ${MODEL_PATH} \
#     && apt-get remove -y wget \
#     && apt-get autoremove -y \
#     && rm -rf /var/lib/apt/lists/*

# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# # Copy application files
# COPY . .

# # Expose the port the app runs on
# EXPOSE 8000

# # Command to run the application
# CMD ["python", "server.py"]


##############################################################################################

# # Use Python 3.9 slim base image
# FROM python:3.9-slim

# # OpenShift-specific user configuration
# RUN useradd -u 1001 -g 0 -m -d /app appuser && \
#     chown -R 1001:0 /app && \
#     chmod -R g=u /app

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV MODEL_PATH=/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# ENV BASE_DIR=/app

# # Install system dependencies (minimal set)
# RUN apt-get update && apt-get install -y \
#     poppler-utils \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*

# # Create directory structure with OpenShift permissions
# # RUN mkdir -p ${BASE_DIR}/{models,cache,chroma_db} && \
# #     chown -R 1001:0 ${BASE_DIR} && \
# #     chmod -R g=u ${BASE_DIR}



# # # Create directory structure with OpenShift permissions
# # RUN mkdir -p /tmp/app/{models,cache,chroma_db} && \
# #     chown -R 1001:0 /tmp/app && \
# #     chmod -R g=u /tmp/app


# # Create directory structure with OpenShift permissions
# RUN mkdir -p /tmp/app/{models,cache,chroma_db} && \
#     chown -R 1001:0 /tmp/app && \
#     chmod -R 775 /tmp/app  # More restrictive than 777


# # Update environment variables
# ENV MODEL_PATH=../tmp/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# ENV BASE_DIR=../tmp/app


# WORKDIR /app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Pre-download embeddings model (OpenShift-friendly)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='${BASE_DIR}/cache')"

# # Copy application files
# COPY . .

# # Switch to non-root user
# USER 1001

# # Expose the port the app runs on
# EXPOSE 8000

# # Command to run the application
# CMD ["python", "server.py"]


################################################################################
# Use Python 3.9 slim base image
FROM python:3.9-slim

# OpenShift user configuration
RUN useradd -u 1001 -g 0 -m -d /app appuser && \
    chown -R 1001:0 /app && \
    chmod -R g=u /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# ENV MODEL_PATH="/app/data/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
ENV MODEL_PATH="/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" 
ENV BASE_DIR="/app/data"
ENV HF_HOME="/app/data/cache"
ENV HUGGINGFACE_HUB_CACHE="/app/data/cache"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p ${BASE_DIR}/{cache,chroma_db} && \
    chown -R 1001:0 ${BASE_DIR} && \
    chmod -R 775 ${BASE_DIR}

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embeddings model
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='${BASE_DIR}/cache')"

# Copy application files
COPY . .

# Switch to non-root user
USER 1001

EXPOSE 8000
CMD ["python", "server.py"]