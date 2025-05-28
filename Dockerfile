FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hard-code port 8080 (Railway default)
CMD ["streamlit", "run", "multilingual_voice_tutor_enhanced.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
