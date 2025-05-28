FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1-dev \
    ffmpeg \
    gcc \
    g++ \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Start command
CMD ["streamlit", "run", "multilingual_voice_tutor_enhanced.py", "--server.port", "8080", "--server.address", "0.0.0.0", "--server.headless", "true", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
