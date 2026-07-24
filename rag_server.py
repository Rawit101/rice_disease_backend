"""
rag_server.py - RAG Search Server สำหรับค้นหาข้อมูลโรคข้าวด้วย Semantic Search

วิธีใช้:
    python rag_server.py

Endpoints:
    POST /search  - ค้นหา context ที่เกี่ยวข้อง (ไม่ generate คำตอบ)
    POST /query   - ค้นหา context + generate คำตอบด้วย Gemini
    GET  /health  - Health check
    GET  /stats   - ดูสถิติของ knowledge base

ต้องการ:
    - GEMINI_API_KEY ใน .env
    - chroma_db/ (สร้างด้วย python scripts/build_rag.py)
"""

import os
import sys
import io
import time

# แก้ปัญหา Unicode encoding บน Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import chromadb
    from google import genai
except ImportError as e:
    print(f"ERROR: missing dependency: {e}")
    print("Please install first:")
    print("  pip install flask flask-cors chromadb google-genai python-dotenv")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, '.env'))

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    sys.exit(1)

MAXPLUS_API_KEY = os.getenv('MAXPLUS_API_KEY')
MAXPLUS_BASE_URL = os.getenv('MAXPLUS_BASE_URL', 'https://api.maxplus-ai.cc')

# ใช้ google.genai (ตัวใหม่) แทน google.generativeai (deprecated)
client = genai.Client(api_key=GEMINI_API_KEY)

CHROMA_DIR = os.path.join(ROOT_DIR, 'chroma_db')
COLLECTION_NAME = 'rice_diseases'
EMBEDDING_MODEL = 'gemini-embedding-001'
TOP_K = 3  # จำนวน chunks ที่ดึงมาใช้เป็น context

# =====================
# Flask App
# =====================
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# =====================
# โหลด ChromaDB
# =====================
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"[OK] ChromaDB loaded: {collection.count()} chunks in '{COLLECTION_NAME}'")
except Exception as e:
    print(f"[ERROR] Cannot load ChromaDB: {e}")
    print("Please build knowledge base first: python scripts/build_rag.py")
    collection = None


def create_query_embedding(text):
    """สร้าง embedding สำหรับคำถาม (ใช้ task_type ต่างจาก document)"""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={'task_type': 'RETRIEVAL_QUERY'}
    )
    return result.embeddings[0].values


def search_knowledge(question, disease_hint='', top_k=TOP_K):
    """
    ค้นหา context ที่เกี่ยวข้องจาก ChromaDB ด้วย semantic search

    Args:
        question: คำถามจาก user
        disease_hint: ชื่อโรคจากผลวิเคราะห์ YOLO (ถ้ามี)
        top_k: จำนวน chunks ที่ต้องการ

    Returns:
        list of dict: [{ text, title, source, score }]
    """
    if collection is None:
        return []

    # รวม question + disease_hint เพื่อให้ค้นหาได้แม่นยำขึ้น
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

            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # แปลงเป็น similarity score (0-1)
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
# API Endpoints
# =====================

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'collection_loaded': collection is not None,
        'total_chunks': collection.count() if collection else 0
    })


@app.route('/stats')
def stats():
    if collection is None:
        return jsonify({'error': 'Collection not loaded'}), 500

    # ดึง metadata ทั้งหมดเพื่อแสดงสถิติ
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


@app.route('/search', methods=['POST'])
def search():
    """
    ค้นหา context ที่เกี่ยวข้อง (search only - ไม่ generate คำตอบ)

    Request body:
        {
            "question": "ใบข้าวเป็นจุดสีน้ำตาล",
            "disease_hint": "โรคใบจุดสีน้ำตาล",  // optional
            "top_k": 3  // optional, default 3
        }

    Response:
        {
            "results": [...],
            "context": "combined context string for prompt injection"
        }
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing "question" field'}), 400

    question = data['question']
    disease_hint = data.get('disease_hint', '')
    top_k = min(data.get('top_k', TOP_K), 10)  # จำกัดสูงสุด 10

    start_time = time.time()
    results = search_knowledge(question, disease_hint, top_k)
    search_time = round((time.time() - start_time) * 1000)  # ms

    # สร้าง context string สำหรับใส่ใน Gemini prompt
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
    """
    ค้นหา context + generate คำตอบด้วย Gemini (full RAG)

    Request body:
        {
            "question": "โรคใบไหม้แก้ยังไง",
            "disease_hint": "",
            "top_k": 3
        }

    Response:
        {
            "answer": "...",
            "sources": [...],
            "search_time_ms": 120
        }
    """
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing "question" field'}), 400

    question = data['question']
    disease_hint = data.get('disease_hint', '')
    top_k = min(data.get('top_k', TOP_K), 10)

    # Step 1: ค้นหา context
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

    # Step 2: สร้าง context string
    context_parts = []
    for r in results:
        context_parts.append(r['text'])
    context = '\n\n---\n\n'.join(context_parts)

    # Step 3: Generate คำตอบด้วย MaxPlus AI (Claude/GPT)
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
            raise ValueError("MAXPLUS_API_KEY is not set in .env")
            
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


# =====================
# Start Server
# =====================
if __name__ == '__main__':
    print("=" * 50)
    print("RAG Search Server Starting")
    print("=" * 50)
    print(f"  ChromaDB: {CHROMA_DIR}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total chunks: {collection.count() if collection else 0}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5001, debug=False)
