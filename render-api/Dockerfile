FROM python:3.11-slim

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (NO PyTorch — only onnxruntime!)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py ./
COPY best.onnx ./
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

EXPOSE 5000

CMD ["python", "app.py"]
