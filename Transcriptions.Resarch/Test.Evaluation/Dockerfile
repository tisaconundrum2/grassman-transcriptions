FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir 'datasets<4' librosa soundfile jiwer

# Copy shared code + both training model folders (eval needs either) + eval app
COPY Transcriptions.Resarch/Shared/ Shared/
COPY Transcriptions.Resarch/Train.Ablation/ Train/Ablation/
COPY Transcriptions.Resarch/Train.Grassman/ Train/Grassman/
COPY Transcriptions.Resarch/Test.Evaluation/ Test.Evaluation/

# No CMD — command is specified at launch
# Example:
#   python -u Test.Evaluation/app.py
