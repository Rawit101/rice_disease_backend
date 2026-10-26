FROM python:3.11-slim

# Install system dependencies for OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (PyTorch CPU + ultralytics)
COPY render-api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Optional compatibility for standalone API consumers, never required by LINE.
ARG ENABLE_RAG=false
ENV ENABLE_RAG=${ENABLE_RAG}
RUN if [ "$ENABLE_RAG" = "true" ]; then pip install --no-cache-dir 'chromadb>=0.4.0' 'google-genai>=1.0.0'; fi

# Copy application files
COPY render-api/app.py ./render-api/app.py
COPY best.pt ./
COPY chroma_db/ ./chroma_db/
COPY data/ ./data/

EXPOSE 5000

CMD ["python", "render-api/app.py"]
