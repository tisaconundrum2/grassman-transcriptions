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
# Pin datasets to version that supports soundfile decoder (newer versions require torchcodec + libnvrtc)
# librosa needed by datasets<4 for audio decoding, soundfile for the backend
RUN pip install --no-cache-dir 'datasets<4' librosa soundfile

# Copy project
COPY src/ src/
COPY app.py .
COPY train_gpu.py .
COPY export_onnx.py .
COPY inference.py .

# Conformer-CTC: Grassmann + ConvModule + two-sided mixing
# 3K samples to fit in T4's 16GB system RAM, 150 epochs to compensate
# Batch size 8, max audio 15s to reduce padding waste
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CMD ["python", "-u", "train_gpu.py", \
    "--max-samples", "3000", \
    "--epochs", "150", \
    "--batch-size", "8", \
    "--layers", "6", \
    "--d-model", "256", \
    "--r-reduced", "32", \
    "--lr", "1.5e-4", \
    "--ctc", \
    "--conformer", \
    "--two-sided", \
    "--save-dir", "checkpoints", \
    "--decode-every", "5"]
