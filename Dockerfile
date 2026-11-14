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

# Copy project
COPY src/ src/
COPY app.py .
COPY train_gpu.py .
COPY export_onnx.py .
COPY inference.py .

# CTC mode: no repetition collapse, 5000 samples, 50 epochs on T4
# Estimated: ~30-60 min on T4, cost ~$0.20-0.40
CMD ["python", "-u", "train_gpu.py", \
    "--max-samples", "5000", \
    "--epochs", "50", \
    "--batch-size", "32", \
    "--layers", "6", \
    "--d-model", "256", \
    "--r-reduced", "32", \
    "--decoder-layers", "4", \
    "--lr", "3e-4", \
    "--ctc", \
    "--save-dir", "checkpoints", \
    "--decode-every", "10"]
