FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 portaudio19-dev gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
