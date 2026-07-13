# 🌾 Rice Disease Backend — LINE Bot "ไอนาย"

> ระบบ Backend สำหรับ LINE Bot ผู้เชี่ยวชาญด้านโรคข้าว ที่ใช้ **YOLOv8** วิเคราะห์โรคจากรูปใบข้าว และ **Gemini AI** ตอบคำถามเกี่ยวกับการดูแลข้าว พร้อมระบบ **RAG (Retrieval-Augmented Generation)** ที่ดึงข้อมูลจากฐานความรู้กรมการข้าว

---

## 📋 สารบัญ

- [ฟีเจอร์หลัก](#-ฟีเจอร์หลัก)
- [สถาปัตยกรรมระบบ](#-สถาปัตยกรรมระบบ)
- [เทคโนโลยีที่ใช้](#-เทคโนโลยีที่ใช้)
- [โครงสร้างโปรเจกต์](#-โครงสร้างโปรเจกต์)
- [ข้อกำหนดเบื้องต้น](#-ข้อกำหนดเบื้องต้น)
- [การติดตั้ง](#-การติดตั้ง)
- [การตั้งค่า Environment Variables](#-การตั้งค่า-environment-variables)
- [การรันระบบ](#-การรันระบบ)
- [API Endpoints](#-api-endpoints)
- [โรคข้าวที่รองรับ](#-โรคข้าวที่รองรับ)
- [Rich Menu](#-rich-menu)
- [ฐานข้อมูล](#-ฐานข้อมูล)

---

## ✨ ฟีเจอร์หลัก

| ฟีเจอร์ | รายละเอียด |
|---------|-----------|
| 📸 **วิเคราะห์โรคข้าวด้วย AI** | ส่งรูปใบข้าวผ่าน LINE → YOLOv8 ตรวจจับโรค → ส่งรูป annotated + คำแนะนำกลับ |
| 💬 **แชทบอทถาม-ตอบ** | ถามเรื่องโรคข้าว การปลูก ปุ๋ย ยา ผ่าน Gemini AI พร้อมบริบทจากผลวิเคราะห์ |
| 🔍 **RAG Semantic Search** | ค้นหาข้อมูลโรคข้าวจากฐานความรู้กรมการข้าว ด้วย ChromaDB + Gemini Embeddings |
| 📝 **Keyword Search Fallback** | ถ้า RAG Server ไม่พร้อม ระบบจะ fallback ไปใช้ keyword search จากไฟล์ JSON |
| 🗄️ **บันทึกประวัติ** | เก็บข้อมูลผู้ใช้ ผลวิเคราะห์ และประวัติแชทใน SQLite |
| 🔐 **Webhook Security** | LINE Signature Verification + Rate Limiting |
| 📊 **Dashboard API** | REST API สำหรับดึงสถิติการใช้งาน |
| 🖼️ **Rich Menu** | เมนูลัดในแชท LINE สำหรับเข้าถึงฟีเจอร์ต่าง ๆ |

---

## 🏗️ สถาปัตยกรรมระบบ

```
┌─────────────┐       ┌──────────────────────────────────────────────┐
│  LINE App   │       │              Backend Server                  │
│  (ผู้ใช้)    │◄─────►│                                              │
└─────────────┘       │  ┌──────────────────────────────────────┐    │
                      │  │  index.js (Express, port 3000)       │    │
                      │  │  - LINE Webhook Handler              │    │
                      │  │  - Gemini AI Chatbot                 │    │
                      │  │  - Session Management                │    │
                      │  │  - Knowledge Search (keyword)        │    │
                      │  │  - Dashboard API                     │    │
                      │  └──────┬────────────┬──────────────────┘    │
                      │         │            │                       │
                      │    ┌────▼────┐  ┌────▼────────────┐          │
                      │    │  YOLO   │  │  RAG Server     │          │
                      │    │  Server │  │  (rag_server.py) │          │
                      │    │  :5000  │  │  :5001           │          │
                      │    └────┬────┘  └────┬────────────┘          │
                      │         │            │                       │
                      │    ┌────▼────┐  ┌────▼────┐  ┌──────────┐   │
                      │    │best.pt  │  │ChromaDB │  │ SQLite   │   │
                      │    │(YOLOv8) │  │(vectors)│  │(database)│   │
                      │    └─────────┘  └─────────┘  └──────────┘   │
                      └──────────────────────────────────────────────┘
```

---

## 🛠️ เทคโนโลยีที่ใช้

### Node.js (Main Server)

| Package | หน้าที่ |
|---------|--------|
| `express` | Web framework สำหรับ Webhook + REST API |
| `axios` | HTTP client สำหรับเรียก LINE API, YOLO, RAG |
| `@google/generative-ai` | Gemini API สำหรับ chatbot |
| `better-sqlite3` | SQLite database |
| `dotenv` | จัดการ environment variables |
| `form-data` | ส่งรูปไปยัง YOLO Server |

### Python (ML Servers)

| Package | หน้าที่ |
|---------|--------|
| `flask` + `flask-cors` | API Server สำหรับ YOLO และ RAG |
| `ultralytics` | YOLOv8 object detection |
| `opencv-python` | ประมวลผลภาพ, วาด bounding box |
| `chromadb` | Vector database สำหรับ semantic search |
| `google-genai` | Gemini Embeddings สำหรับ RAG |
| `torch` + `torchvision` | Deep Learning runtime |

---

## 📁 โครงสร้างโปรเจกต์

```
backend-farmer/
├── index.js                  # 🚀 Main server — LINE Webhook + Chatbot + API
├── database.js               # 🗄️ SQLite module — Users, Analyses, Chat History
├── yolo_server.py            # 🤖 YOLO API Server — วิเคราะห์โรคจากรูปภาพ
├── rag_server.py             # 🔍 RAG Server — Semantic search จาก ChromaDB
├── setup-richmenu.js         # 🖼️ สคริปต์สร้าง Rich Menu บน LINE
├── combine_richmenu.py       # 🎨 สคริปต์รวมรูปเป็น Rich Menu image
├── package.json              # Node.js dependencies
├── requirements.txt          # Python dependencies
├── .env                      # 🔐 Environment variables (ไม่ commit)
├── .gitignore
│
├── best.pt                   # 🧠 YOLO model weights (ไม่ commit)
│
├── scripts/
│   ├── build-knowledge.js    # Scrape ข้อมูลโรคข้าวจาก Rice Knowledge Bank
│   └── build_rag.py          # สร้าง embeddings → ChromaDB
│
├── data/
│   ├── rice_knowledge.json   # ข้อมูลโรคข้าว (scraped)
│   └── rice_farmer.db        # SQLite database
│
├── chroma_db/                # ChromaDB vector storage
├── public/results/           # รูปผลวิเคราะห์ (ชั่วคราว, ลบอัตโนมัติ)
└── temp/                     # ไฟล์ชั่วคราว
```

---

## 📋 ข้อกำหนดเบื้องต้น

- **Node.js** v18+
- **Python** 3.9+
- **LINE Developer Account** — สร้าง Messaging API Channel
- **Google AI API Key** — สำหรับ Gemini AI
- **ngrok** หรือ domain สาธารณะ — สำหรับ LINE Webhook


---

## 📡 API Endpoints

### LINE Webhook

| Method | Path | รายละเอียด |
|--------|------|-----------|
| `POST` | `/webhook` | LINE Webhook — รับ events จาก LINE Platform |

### REST APIs

| Method | Path | รายละเอียด |
|--------|------|-----------|
| `GET` | `/` | Health check |
| `GET` | `/api/stats` | สถิติรวมของระบบ (จำนวนผู้ใช้, วิเคราะห์, โรคที่พบบ่อย) |
| `GET` | `/api/users/:userId/history` | ประวัติการใช้งานของผู้ใช้ |

### YOLO Server (port 5000)

| Method | Path | รายละเอียด |
|--------|------|-----------|
| `GET` | `/` | สถานะ server + model info |
| `GET` | `/health` | Health check |
| `GET` | `/classes` | รายชื่อ class ที่โมเดลรองรับ |
| `POST` | `/predict` | วิเคราะห์โรคจากรูปภาพ (multipart/form-data, key: `image`) |

### RAG Server (port 5001)

| Method | Path | รายละเอียด |
|--------|------|-----------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | สถิติ knowledge base |
| `POST` | `/search` | Semantic search (search only) |
| `POST` | `/query` | Search + Generate คำตอบด้วย Gemini |

---

## 🦠 โรคข้าวที่รองรับ

โมเดล YOLOv8 (`best.pt`) สามารถตรวจจับโรคข้าวได้ดังนี้:

| # | ชื่อโรค (ไทย) | ชื่อโรค (English) | Class Name |
|---|--------------|------------------|------------|
| 1 | โรคขอบใบแห้ง | Bacterial Leaf Blight | `bacterial_leaf_blight` |
| 2 | โรคใบจุดสีน้ำตาล | Brown Spot | `brown_spot` |
| 3 | โรคไหม้ / โรคใบไหม้ | Rice Blast / Leaf Blast | `rice_blast` / `leaf_blast` |
| 4 | โรคใบขีดสีน้ำตาล | Narrow Brown Leaf Spot | `narrow_brown_leaf_spot` |
| 5 | โรคดอกกระถิน | False Smut | `false_smut` |
| 6 | โรคเมล็ดด่าง | Dirty Seed | `dirty_seed` |
| 7 | โรคกาบใบเน่า | Sheath Rot | `sheath_rot` |
| 8 | โรคลำต้นเน่า | Stem Rot | `stem_rot` |
| 9 | โรคใบแถบแดง | Red Stripe | `red_stripe` |
| 10 | ไม่พบโรค (สุขภาพดี) | Healthy | `healthy` |

---

## 🖼️ Rich Menu

ระบบมี Rich Menu 3 ปุ่ม (layout แนวตั้ง 2500×1686 px):

| ปุ่ม | ฟังก์ชัน |
|------|---------|
| 🔝 **องค์ความรู้เรื่องข้าว** | เปิดเว็บ [Rice Knowledge Bank](https://rkb.ricethailand.go.th) |
| 📸 **เลือกรูปภาพ** | เปิด Camera Roll เพื่อส่งรูปวิเคราะห์โรค |
| 🦠 **โรคข้าว** | เปิดเว็บแสดงข้อมูลโรคข้าว |


---

## 🗄️ ฐานข้อมูล

ใช้ **SQLite** (`data/rice_farmer.db`) เก็บข้อมูล 3 ตาราง:

### `users` — ข้อมูลผู้ใช้ LINE

| Column | Type | รายละเอียด |
|--------|------|-----------|
| `line_user_id` | TEXT (UNIQUE) | LINE User ID |
| `display_name` | TEXT | ชื่อที่แสดงใน LINE |
| `first_seen_at` | DATETIME | เวลาที่เห็นครั้งแรก |
| `last_active_at` | DATETIME | เวลาที่ใช้งานล่าสุด |
| `total_analyses` | INTEGER | จำนวนครั้งที่วิเคราะห์โรค |
| `total_messages` | INTEGER | จำนวนข้อความที่ส่ง |

### `analyses` — ผลการวิเคราะห์โรคข้าว

| Column | Type | รายละเอียด |
|--------|------|-----------|
| `line_user_id` | TEXT | LINE User ID |
| `disease` | TEXT | ชื่อโรคที่พบ |
| `confidence` | REAL | ค่าความมั่นใจ (0-1) |
| `severity` | TEXT | ระดับความรุนแรง (ต่ำ/ปานกลาง/สูง) |
| `advice` | TEXT | คำแนะนำจาก Gemini AI |
| `image_url` | TEXT | URL รูปที่วาด bounding box |

### `chat_history` — ประวัติการสนทนา

| Column | Type | รายละเอียด |
|--------|------|-----------|
| `line_user_id` | TEXT | LINE User ID |
| `role` | TEXT | `user` หรือ `bot` |
| `message` | TEXT | เนื้อหาข้อความ |
| `created_at` | DATETIME | เวลาที่ส่ง |

---

## ⚙️ การทำงานภายใน

### Flow การวิเคราะห์โรคจากรูปภาพ

```
1. ผู้ใช้ส่งรูปใบข้าวผ่าน LINE
2. Webhook รับ event → ตอบ "กำลังวิเคราะห์..."
3. ดาวน์โหลดรูปจาก LINE Content API
4. ส่งรูปไปยัง YOLO Server (/predict)
5. YOLO วิเคราะห์ → ส่งผลลัพธ์ + รูป annotated กลับ
6. เรียก Gemini AI สร้างคำแนะนำ (หรือใช้ fallback ถ้า API ล่ม)
7. บันทึกรูป annotated → public/results/
8. บันทึกผลลง SQLite + Session (in-memory)
9. Push ข้อความ + รูปผลลัพธ์กลับไปยังผู้ใช้ผ่าน LINE
```

### Flow การตอบคำถาม (Chat)

```
1. ผู้ใช้พิมพ์คำถามเรื่องข้าว
2. ตรวจจับทักทาย/ลา/ขอบคุณ → ตอบทันที (ไม่เรียก AI)
3. ลอง RAG Semantic Search ก่อน (ChromaDB)
4. ถ้า RAG ไม่พร้อม → Fallback เป็น Keyword Search
5. รวม context (ผลวิเคราะห์ + ประวัติสนทนา + ข้อมูลโรค) → สร้าง prompt
6. เรียก Gemini AI (ลอง gemini-2.5-flash-lite ก่อน → fallback เป็น gemini-2.5-flash)
7. บันทึกประวัติแชท + ตอบผู้ใช้ผ่าน LINE
```

---

## 📝 License

This project is for educational purposes.

---

## 👨‍💻 ผู้พัฒนา

พัฒนาโดยทีมนักศึกษาเพื่อช่วยเหลือเกษตรกรชาวนาไทย 🌾
