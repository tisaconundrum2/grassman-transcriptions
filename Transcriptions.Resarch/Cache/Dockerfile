FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir 'datasets<4' librosa soundfile

# Copy shared code + cache app
COPY Transcriptions.Resarch/Shared/ Shared/
COPY Transcriptions.Resarch/Cache/ Cache/

# No CMD — command is specified at launch
# Example:
#   python -u Cache/app.py
