FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir onnxruntime onnxscript
RUN pip install --no-cache-dir 'datasets<4' librosa soundfile azure-storage-blob

# Copy shared code + this training app
COPY Transcriptions.Resarch/Shared/ Shared/
COPY Transcriptions.Resarch/Train.Ablation/ Train.Ablation/

# No CMD — command is specified at launch via `hf jobs run`
# Example:
#   hf jobs run --flavor t4-small ghcr.io/tisaconundrum2/grassmann-ablation:<sha> \
#     -- python -u Train.Ablation/app.py
