FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Use soundfile for audio decoding (torchcodec needs libnvrtc which isn't in the runtime image)
ENV DATASETS_AUDIO_DECODER=soundfile

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

# Conformer-CTC: Grassmann + ConvModule + two-sided mixing
# 5000 samples, 50 epochs on T4. Estimated cost ~$0.20-0.40
CMD ["python", "-u", "train_gpu.py", \
    "--max-samples", "5000", \
    "--epochs", "50", \
    "--batch-size", "32", \
    "--layers", "6", \
    "--d-model", "256", \
    "--r-reduced", "32", \
    "--lr", "3e-4", \
    "--ctc", \
    "--conformer", \
    "--two-sided", \
    "--save-dir", "checkpoints", \
    "--decode-every", "10"]
