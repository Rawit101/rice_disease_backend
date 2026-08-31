"""
app.py - Combined YOLO (ONNX) + RAG Server สำหรับ Render Deploy
รวม yolo_onnx_server.py + rag_server.py เป็น API เดียว

Endpoints:
    POST /predict  - YOLO rice disease detection (ONNX)
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of render-api/)

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort
import chromadb
from dotenv import load_dotenv

# Safe import for Gemini
try:
    from google import genai
    USE_NEW_GENAI = True
except Exception:
    try:
        import google.generativeai as genai
        USE_NEW_GENAI = False
    except Exception as e:
        genai = None
        USE_NEW_GENAI = False
        print(f"⚠️ Warning: Could not import genai: {e}")

load_dotenv(os.path.join(ROOT_DIR, '.env'))

# =====================
# Flask App
# =====================
app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# =====================
# YOLO ONNX Model
# =====================
MODEL_PATH = os.path.join(ROOT_DIR, "best.onnx")
CLASS_NAMES = {}
session = None
# Limit OpenCV threads for cloud container
cv2.setNumThreads(1)

try:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    metadata = session.get_modelmeta().custom_metadata_map
    if 'names' in metadata:
        import ast
        CLASS_NAMES = ast.literal_eval(metadata['names'])

    print(f"✅ YOLO ONNX Model loaded from {MODEL_PATH}")
    print(f"📋 Classes: {CLASS_NAMES}")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")

# =====================
# RAG / ChromaDB
# =====================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MAXPLUS_API_KEY = os.getenv('MAXPLUS_API_KEY')
MAXPLUS_BASE_URL = os.getenv('MAXPLUS_BASE_URL', 'https://api.maxplus-ai.cc')

gemini_client = None
collection = None

CHROMA_DIR = os.path.join(ROOT_DIR, 'chroma_db')
COLLECTION_NAME = 'rice_diseases'
EMBEDDING_MODEL = 'gemini-embedding-001'
TOP_K = 3

if GEMINI_API_KEY and genai:
    try:
        if USE_NEW_GENAI:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_client = genai
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"✅ ChromaDB loaded: {collection.count()} chunks in '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"⚠️ ChromaDB not loaded: {e}")
else:
    print("⚠️ GEMINI_API_KEY not set or genai unavailable — RAG endpoints disabled")


# =====================
# YOLO Helper Functions
# =====================
def preprocess_image(pil_img, target_size=640):
    """Preprocess image สำหรับ YOLO ONNX inference"""
    img = np.array(pil_img)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    orig_h, orig_w = img.shape[:2]

    scale = min(target_size / orig_h, target_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    return blob, orig_w, orig_h, scale, left, top


def postprocess_detections(output, orig_w, orig_h, scale, pad_left, pad_top, conf_threshold=0.05):
    """Postprocess YOLO ONNX output (NMS format) เป็น detections
    Format: [x1, y1, x2, y2, confidence, class_id, mask_coeffs...]"""
    predictions = []
    det = output[0]

    if det.ndim == 3:
        det = det[0]

    for row in det:
        # NMS format: [x1, y1, x2, y2, conf, class_id, ...]
        bx1, by1, bx2, by2 = row[0], row[1], row[2], row[3]
        confidence = float(row[4])
        class_id = int(row[5])

        if confidence < conf_threshold:
            continue

        x1 = (bx1 - pad_left) / scale
        y1 = (by1 - pad_top) / scale
        x2 = (bx2 - pad_left) / scale
        y2 = (by2 - pad_top) / scale

        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))

        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        predictions.append({
            "class": class_name,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return predictions


def draw_annotations(img, predictions):
    """วาด bounding boxes บนรูป"""
    annotated = img.copy()
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
    ]
    for pred in predictions:
        x1, y1, x2, y2 = [int(v) for v in pred['bbox']]
        class_name = pred['class']
        conf = pred['confidence']
        color = colors[pred['class_id'] % len(colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated


# =====================
# RAG Helper Functions
# =====================
def create_query_embedding(text):
    """สร้าง embedding สำหรับคำถาม"""
    if USE_NEW_GENAI:
        result = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        return result.embeddings[0].values
    else:
        # Fallback to google.generativeai
        model_name = f"models/{EMBEDDING_MODEL}" if not EMBEDDING_MODEL.startswith("models/") else EMBEDDING_MODEL
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']


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
        "runtime": "ONNX (lightweight, no PyTorch)",
        "services": {
            "yolo": {"loaded": session is not None, "classes": CLASS_NAMES},
            "rag": {"loaded": collection is not None, "chunks": collection.count() if collection else 0}
        }
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "yolo_model_loaded": session is not None,
        "rag_collection_loaded": collection is not None
    })


# ---------- YOLO Endpoints ----------

@app.route('/predict', methods=['POST'])
def predict():
    if session is None:
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
            pil_img = Image.open(image_stream).convert('RGB')
            print(f"📸 Original image size: {pil_img.size}")
        except Exception as e:
            return jsonify({"error": "Invalid image", "message": f"Cannot decode image: {str(e)}"}), 400

        t0 = time.time()
        blob, orig_w, orig_h, scale, pad_left, pad_top = preprocess_image(pil_img)
        t_prep = time.time()
        
        outputs = session.run(None, {input_name: blob})
        t_infer = time.time()
        
        predictions = postprocess_detections(
            outputs, orig_w, orig_h, scale, pad_left, pad_top, conf_threshold=0.05
        )
        t_post = time.time()

        print(f"⏱️ Prep: {int((t_prep-t0)*1000)}ms | Infer: {int((t_infer-t_prep)*1000)}ms | Post: {int((t_post-t_infer)*1000)}ms")
        print(f"✅ Found {len(predictions)} predictions")

        annotated_image_base64 = None
        if len(predictions) > 0:
            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            annotated_img = draw_annotations(img_bgr, predictions)
            _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


@app.route('/classes')
def get_classes():
    if session is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({"classes": CLASS_NAMES, "total": len(CLASS_NAMES)})


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

    prompt = f"""คุณคือผู้เชี่ยวชาญด้านโรคข้าว ชื่อ "ไอนาย"
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
    print("🌾 Rice Disease API (YOLO ONNX + RAG)")
    print("=" * 50)
    print(f"  YOLO Model: {'✅ loaded (ONNX)' if session else '❌ not loaded'}")
    print(f"  RAG/ChromaDB: {'✅ loaded' if collection else '❌ not loaded'}")
    if CLASS_NAMES:
        print(f"  Classes: {CLASS_NAMES}")
    if collection:
        print(f"  Chunks: {collection.count()}")
    print("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
