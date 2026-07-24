"""
build_rag.py - สร้าง embeddings จาก rice_knowledge.json แล้วบันทึกลง ChromaDB

วิธีใช้:
    python scripts/build_rag.py

ต้องการ:
    - GEMINI_API_KEY ใน .env
    - data/rice_knowledge.json (สร้างด้วย npm run build:knowledge)
"""

import json
import os
import sys
import time
import re
import io

# แก้ปัญหา Unicode encoding บน Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# เพิ่ม root directory เข้า path เพื่อให้อ่าน .env ได้
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

try:
    import chromadb
    from google import genai
except ImportError as e:
    print(f"ERROR: missing dependency: {e}")
    print("Please install first:")
    print("  pip install chromadb google-genai python-dotenv")
    sys.exit(1)

from dotenv import load_dotenv

# โหลด .env จาก root directory
load_dotenv(os.path.join(ROOT_DIR, '.env'))

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    sys.exit(1)

# ใช้ google.genai (ตัวใหม่) แทน google.generativeai (deprecated)
client = genai.Client(api_key=GEMINI_API_KEY)

KNOWLEDGE_FILE = os.path.join(ROOT_DIR, 'data', 'rice_knowledge.json')
CHROMA_DIR = os.path.join(ROOT_DIR, 'chroma_db')
COLLECTION_NAME = 'rice_diseases'
EMBEDDING_MODEL = 'gemini-embedding-001'
CHUNK_MAX_CHARS = 800  # ขนาดสูงสุดของแต่ละ chunk


def load_knowledge():
    """โหลดข้อมูลโรคข้าวจาก JSON"""
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"ERROR: File not found: {KNOWLEDGE_FILE}")
        print("Build it first: npm run build:knowledge")
        sys.exit(1)

    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = data.get('entries', [])
    source = data.get('source', 'unknown')
    print(f"[INFO] Loaded {len(entries)} entries from {source}")
    return entries


def chunk_text(text, title, source, max_chars=CHUNK_MAX_CHARS):
    """
    แบ่งเนื้อหาเป็น chunks ที่มีขนาดเหมาะสม
    แต่ละ chunk จะมี title กำกับเพื่อให้ค้นหาได้ง่าย
    """
    # ล้าง whitespace ซ้ำ
    text = re.sub(r'\n{3,}', '\n\n', text.strip())

    # ถ้าสั้นพอ ใช้เป็น 1 chunk
    if len(text) <= max_chars:
        return [{
            'text': f"{title}\n\n{text}",
            'title': title,
            'source': source,
            'chunk_index': 0
        }]

    # แบ่งตาม section (ขึ้นบรรทัดใหม่ 2 ครั้ง)
    sections = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # ถ้าเพิ่ม section นี้แล้วยังไม่เกิน max_chars -> รวม
        if len(current_chunk) + len(section) + 2 <= max_chars:
            current_chunk = f"{current_chunk}\n\n{section}" if current_chunk else section
        else:
            # บันทึก chunk ปัจจุบัน
            if current_chunk:
                chunks.append({
                    'text': f"{title}\n\n{current_chunk}",
                    'title': title,
                    'source': source,
                    'chunk_index': chunk_index
                })
                chunk_index += 1
            current_chunk = section

    # อย่าลืม chunk สุดท้าย
    if current_chunk:
        chunks.append({
            'text': f"{title}\n\n{current_chunk}",
            'title': title,
            'source': source,
            'chunk_index': chunk_index
        })

    return chunks


def create_embedding(text, retries=3):
    """สร้าง embedding ด้วย Gemini API (google.genai) พร้อม retry"""
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config={'task_type': 'RETRIEVAL_DOCUMENT'}
            )
            return result.embeddings[0].values
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"  WARNING: Embedding failed (attempt {attempt + 1}): {e}")
                print(f"  Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e


def build_knowledge_base():
    """สร้าง knowledge base ใน ChromaDB"""
    entries = load_knowledge()

    # สร้าง chunks จากทุก entry
    all_chunks = []
    for entry in entries:
        title = entry.get('title', '')
        content = entry.get('content', '')
        source = entry.get('source', '')
        if not title or not content:
            continue
        chunks = chunk_text(content, title, source)
        all_chunks.extend(chunks)

    print(f"[INFO] Split into {len(all_chunks)} chunks")

    # ลบ ChromaDB collection เก่า แล้วสร้างใหม่
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # ลบ collection เดิมถ้ามี
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Deleted old collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # ใช้ cosine similarity
    )

    # สร้าง embeddings แล้วบันทึกลง ChromaDB
    print(f"\n[INFO] Creating embeddings ({len(all_chunks)} chunks)...")

    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []

    for i, chunk in enumerate(all_chunks):
        print(f"  [{i + 1}/{len(all_chunks)}] {chunk['title']} (chunk {chunk['chunk_index']})")

        embedding = create_embedding(chunk['text'])

        batch_ids.append(f"chunk_{i}")
        batch_embeddings.append(embedding)
        batch_documents.append(chunk['text'])
        batch_metadatas.append({
            'title': chunk['title'],
            'source': chunk['source'],
            'chunk_index': chunk['chunk_index']
        })

        # Gemini Embedding free tier รองรับ 1500 req/min
        # เพิ่ม delay เล็กน้อยเพื่อความปลอดภัย
        if (i + 1) % 50 == 0:
            time.sleep(1)

    # บันทึกทั้ง batch
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )

    print(f"\n[SUCCESS] Knowledge base built!")
    print(f"  ChromaDB: {CHROMA_DIR}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total chunks: {collection.count()}")


if __name__ == '__main__':
    build_knowledge_base()
