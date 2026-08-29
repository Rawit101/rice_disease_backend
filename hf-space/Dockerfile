FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements-hf.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py ./
COPY best.pt ./
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

# Hugging Face Spaces uses port 7860
EXPOSE 7860

CMD ["python", "app.py"]
