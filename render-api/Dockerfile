FROM python:3.11-slim

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (PyTorch CPU + ultralytics)
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py ./
COPY best.pt ./
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

EXPOSE 5000

CMD ["python", "app.py"]
