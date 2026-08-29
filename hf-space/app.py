"""
app.py - Combined YOLO + RAG Server for Hugging Face Spaces
รวม yolo_server.py + rag_server.py เป็น API เดียว

Endpoints:
    POST /predict  - YOLO rice disease detection
    POST /search   - RAG semantic search
    POST /query    - RAG search + AI answer
    GET  /health   - Health check
    GET  /         - Status page
"""

import os
import sys
import io
import time
import base64

# Fix Unicode encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, '.env'))

# =====================
# Flask App
# =====================
app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# =====================
# YOLO Model
# =====================
MODEL_PATH = os.path.join(ROOT_DIR, "best.pt")

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO Model loaded from {MODEL_PATH}")
    print(f"📋 Classes: {model.names}")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# =====================
# RAG / ChromaDB
# =====================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MAXPLUS_API_KEY = os.getenv('MAXPLUS_API_KEY')
MAXPLUS_BASE_URL = os.getenv('MAXPLUS_BASE_URL', 'https://api.maxplus-ai.cc')

client = None
collection = None

CHROMA_DIR = os.path.join(ROOT_DIR, 'chroma_db')
COLLECTION_NAME = 'rice_diseases'
EMBEDDING_MODEL = 'gemini-embedding-001'
TOP_K = 3

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB loaded: {collection.count()} chunks in '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"⚠️ ChromaDB not loaded: {e}")
else:
    print("⚠️ GEMINI_API_KEY not set — RAG endpoints disabled")


# =====================
# Helper Functions
# =====================
def create_query_embedding(text):
    """สร้าง embedding สำหรับคำถาม"""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={'task_type': 'RETRIEVAL_QUERY'}
    )
    return result.embeddings[0].values


def search_knowledge(question, disease_hint='', top_k=TOP_K):
    """ค้นหา context ที่เกี่ยวข้องจาก ChromaDB"""
    if collection is None:
        return []

    query_text = f"{question} {disease_hint}".strip()

    try:
        query_embedding = create_query_embedding(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        if not results or not results['documents'] or not results['documents'][0]:
            return []

        search_results = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            distance = results['distances'][0][i] if results['distances'] else 0
            similarity = 1 - (distance / 2)

            search_results.append({
                'text': doc,
                'title': metadata.get('title', ''),
                'source': metadata.get('source', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'similarity': round(similarity, 4)
            })

        return search_results

    except Exception as e:
        print(f"[ERROR] Search error: {e}")
        return []


# =====================
# Endpoints
# =====================

@app.route('/')
def home():
    return jsonify({
        "status": "Rice Disease API is running",
        "services": {
            "yolo": {"loaded": model is not None, "classes": model.names if model else {}},
            "rag": {"loaded": collection is not None, "chunks": collection.count() if collection else 0}
        }
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "yolo_model_loaded": model is not None,
        "rag_collection_loaded": collection is not None
    })


# ---------- YOLO Endpoints ----------

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded", "message": f"Please check if {MODEL_PATH} exists"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided", "message": "Please send image with key 'image'"}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({"error": "Image too large", "message": "Image must be less than 10MB"}), 400

        if len(image_bytes) == 0:
            return jsonify({"error": "Empty image", "message": "Image file is empty"}), 400

        image_stream = io.BytesIO(image_bytes)
        try:
            pil_img = Image.open(image_stream)
            print(f"📸 Original image size: {pil_img.size}")
        except Exception as e:
            return jsonify({"error": "Invalid image", "message": f"Cannot decode image: {str(e)}"}), 400

        results = model.predict(pil_img, conf=0.05)

        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                class_name = model.names[class_id] if class_id in model.names else f"class_{class_id}"
                predictions.append({
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x, 2) for x in bbox]
                })

        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"✅ Found {len(predictions)} predictions")

        annotated_image_base64 = None
        if len(predictions) > 0:
            annotated_img = results[0].plot()
            _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "predictions": predictions,
            "total": len(predictions),
            "image_shape": [pil_img.height, pil_img.width, 3],
            "annotated_image": annotated_image_base64
        })

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


@app.route('/classes')
def get_classes():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({"classes": model.names, "total": len(model.names)})


# ---------- RAG Endpoints ----------

@app.route('/search', methods=['POST'])
def search():
    if collection is None:
        return jsonify({'error': 'RAG not available — GEMINI_API_KEY not set or ChromaDB not loaded'}), 503

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing "question" field'}), 400

    question = data['question']
    disease_hint = data.get('disease_hint', '')
    top_k = min(data.get('top_k', TOP_K), 10)

    start_time = time.time()
    results = search_knowledge(question, disease_hint, top_k)
    search_time = round((time.time() - start_time) * 1000)

    context_parts = []
    for r in results:
        context_parts.append(
            f"หัวข้อ: {r['title']}\n"
            f"แหล่งข้อมูล: {r['source']}\n"
            f"เนื้อหา:\n{r['text']}"
        )
    context = '\n\n---\n\n'.join(context_parts)

    print(f"[SEARCH] \"{question[:50]}\" -> {len(results)} results ({search_time}ms)")

    return jsonify({
        'results': results,
        'context': context,
        'search_time_ms': search_time,
        'total_results': len(results)
    })


@app.route('/query', methods=['POST'])
def query():
    if collection is None:
        return jsonify({'error': 'RAG not available'}), 503

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing "question" field'}), 400

    question = data['question']
    disease_hint = data.get('disease_hint', '')
    top_k = min(data.get('top_k', TOP_K), 10)

    start_time = time.time()
    results = search_knowledge(question, disease_hint, top_k)
    search_time = round((time.time() - start_time) * 1000)

    if not results:
        return jsonify({
            'answer': '',
            'sources': [],
            'search_time_ms': search_time,
            'fallback': True
        })

    context_parts = []
    for r in results:
        context_parts.append(r['text'])
    context = '\n\n---\n\n'.join(context_parts)

    prompt = f"""คุณคือผู้เชี่ยวชาญด้านโรคข้าว ชื่อ "น้องข้าวสวย" เป็นผู้หญิง ใช้สรรพนาม "หนู" แทนตัวเอง ใช้ "ค่ะ/คะ/นะคะ" แทน "ครับ/นะครับ"
ตอบคำถามโดยอ้างอิงจากข้อมูลด้านล่างเท่านั้น ถ้าข้อมูลไม่เพียงพอให้บอกตรงๆ

ข้อมูลอ้างอิง:
{context}

กฎการตอบ:
- ตอบเฉพาะเรื่องข้าว
- ห้ามใช้ ** หรือ markdown
- สั้นกระชับ ไม่เกิน 120 คำ
- ใช้ emoji นำหน้าหัวข้อ
- ภาษาง่ายๆ เป็นกันเอง

คำถาม: {question}"""

    try:
        import urllib.request
        import json

        if not MAXPLUS_API_KEY:
            raise ValueError("MAXPLUS_API_KEY is not set")

        url = f"{MAXPLUS_BASE_URL}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MAXPLUS_API_KEY}"
        }
        payload = {
            "model": "claude-haiku-4-5-20251001",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
        with urllib.request.urlopen(req, timeout=30) as res:
            response_data = json.loads(res.read().decode('utf-8'))
            answer = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
    except Exception as e:
        print(f"[ERROR] MaxPlus generate error: {e}")
        answer = ''

    sources = [{'title': r['title'], 'source': r['source'], 'similarity': r['similarity']} for r in results]

    print(f"[QUERY] \"{question[:50]}\" -> {len(answer)} chars ({search_time}ms search)")

    return jsonify({
        'answer': answer,
        'sources': sources,
        'search_time_ms': search_time,
        'fallback': False
    })


@app.route('/stats')
def stats():
    if collection is None:
        return jsonify({'error': 'Collection not loaded'}), 500

    all_data = collection.get(include=['metadatas'])
    titles = set()
    for meta in all_data['metadatas']:
        titles.add(meta.get('title', 'unknown'))

    return jsonify({
        'collection': COLLECTION_NAME,
        'total_chunks': collection.count(),
        'total_diseases': len(titles),
        'diseases': sorted(list(titles)),
        'embedding_model': EMBEDDING_MODEL
    })


# =====================
# Start Server
# =====================
if __name__ == '__main__':
    print("=" * 50)
    print("🌾 Rice Disease API (YOLO + RAG)")
    print("=" * 50)
    print(f"  YOLO Model: {'✅ loaded' if model else '❌ not loaded'}")
    print(f"  RAG/ChromaDB: {'✅ loaded' if collection else '❌ not loaded'}")
    if model:
        print(f"  Classes: {model.names}")
    if collection:
        print(f"  Chunks: {collection.count()}")
    print("=" * 50)

    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
