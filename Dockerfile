# Multi-stage build for React + Python
FROM node:18-slim AS frontend-builder

# Build the React component
WORKDIR /app/components/audio_recorder/frontend

# Copy package files
COPY components/audio_recorder/frontend/package.json ./
COPY components/audio_recorder/frontend/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source code
COPY components/audio_recorder/frontend/src ./src
COPY components/audio_recorder/frontend/public ./public

# Build the component
RUN npm run build

# Python stage
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all Python files
COPY *.py ./
COPY components/ ./components/

# Copy built React component from previous stage
COPY --from=frontend-builder /app/components/audio_recorder/frontend/build ./components/audio_recorder/frontend/build

# Expose port
EXPOSE 8080

# Run the app
CMD ["python", "app.py"]
