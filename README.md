# 🌾 Rice Disease Backend — LINE Bot "AI Nai"

> A backend system for a LINE Bot specialized in rice disease diagnosis, using **YOLOv8** to analyze rice leaf images and **Gemini AI** to answer rice farming questions, powered by a **RAG (Retrieval-Augmented Generation)** pipeline built on Thailand's Rice Knowledge Bank.

![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)
![Express](https://img.shields.io/badge/Express-000000?style=for-the-badge&logo=express&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)
![LINE](https://img.shields.io/badge/LINE_API-06C755?style=for-the-badge&logo=line&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Running the System](#-running-the-system)
- [API Endpoints](#-api-endpoints)
- [Supported Rice Diseases](#-supported-rice-diseases)
- [Rich Menu](#-rich-menu)
- [Database Schema](#-database-schema)
- [Internal Workflows](#-internal-workflows)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📸 **AI-Powered Disease Detection** | Send a rice leaf photo via LINE → YOLOv8 detects diseases → Returns annotated image + treatment advice |
| 💬 **AI Chatbot (Q&A)** | Ask about rice diseases, farming, fertilizers, and pesticides — powered by Gemini AI with conversation context |
| 🔍 **RAG Semantic Search** | Retrieves relevant rice disease information from a vector database (ChromaDB + Gemini Embeddings) |
| 📝 **Keyword Search Fallback** | If the RAG server is unavailable, the system falls back to keyword-based search from a local JSON file |
| 🗄️ **Persistent History** | Stores user profiles, analysis results, and chat history in SQLite |
| 🔐 **Webhook Security** | LINE Signature Verification + In-memory Rate Limiting |
| 📊 **Dashboard API** | REST API for retrieving system-wide usage statistics |
| 🖼️ **Rich Menu** | Quick-access menu in LINE chat for key features |

---

## 🏗️ System Architecture

```
┌─────────────┐       ┌──────────────────────────────────────────────┐
│  LINE App   │       │              Backend Server                  │
│   (User)    │◄─────►│                                              │
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

## 🛠️ Tech Stack

### Node.js (Main Server)

| Package | Purpose |
|---------|---------|
| `express` | Web framework for Webhook + REST API |
| `axios` | HTTP client for LINE API, YOLO, and RAG calls |
| `@google/generative-ai` | Gemini API for chatbot responses |
| `better-sqlite3` | Embedded SQLite database |
| `dotenv` | Environment variable management |
| `form-data` | Multipart form uploads to YOLO Server |

### Python (ML Servers)

| Package | Purpose |
|---------|---------|
| `flask` + `flask-cors` | Lightweight API servers for YOLO and RAG |
| `ultralytics` | YOLOv8 object detection engine |
| `opencv-python` | Image processing and bounding box rendering |
| `chromadb` | Vector database for semantic search |
| `google-genai` | Gemini Embeddings for RAG pipeline |
| `torch` + `torchvision` | Deep learning runtime |

---

## 📁 Project Structure

```
backend-farmer/
├── index.js                  # 🚀 Main server — LINE Webhook + Chatbot + API
├── database.js               # 🗄️ SQLite module — Users, Analyses, Chat History
├── yolo_server.py            # 🤖 YOLO API Server — Disease detection from images
├── rag_server.py             # 🔍 RAG Server — Semantic search via ChromaDB
├── setup-richmenu.js         # 🖼️ Script to create and upload LINE Rich Menu
├── combine_richmenu.py       # 🎨 Script to combine images into a Rich Menu image
├── package.json              # Node.js dependencies
├── requirements.txt          # Python dependencies
├── .env                      # 🔐 Environment variables (not committed)
├── .gitignore
│
├── best.pt                   # 🧠 YOLO model weights (not committed)
│
├── scripts/
│   ├── build-knowledge.js    # Scrapes rice disease data from Rice Knowledge Bank
│   └── build_rag.py          # Generates embeddings and stores them in ChromaDB
│
├── data/
│   ├── rice_knowledge.json   # Scraped rice disease knowledge
│   └── rice_farmer.db        # SQLite database file
│
├── chroma_db/                # ChromaDB vector storage
├── public/results/           # Temporary annotated result images (auto-cleaned)
└── temp/                     # Temporary files
```

---

## 📋 Prerequisites

- **Node.js** v18+
- **Python** 3.9+
- **LINE Developer Account** — Create a Messaging API Channel
- **Google AI API Key** — For Gemini AI
- **ngrok** or a public domain — For LINE Webhook URL

---

## 📡 API Endpoints

### LINE Webhook

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhook` | LINE Webhook — Receives events from the LINE Platform |

### REST APIs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/api/stats` | System-wide statistics (users, analyses, top diseases) |
| `GET` | `/api/users/:userId/history` | Usage history for a specific user |

### YOLO Server (port 5000)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Server status + model info |
| `GET` | `/health` | Health check |
| `GET` | `/classes` | List of classes supported by the model |
| `POST` | `/predict` | Analyze a rice leaf image (multipart/form-data, key: `image`) |

### RAG Server (port 5001)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Knowledge base statistics |
| `POST` | `/search` | Semantic search (context retrieval only) |
| `POST` | `/query` | Search + Generate answer with Gemini |

---

## 🦠 Supported Rice Diseases

The YOLOv8 model (`best.pt`) can detect the following rice diseases:

| # | Disease (Thai) | Disease (English) | Class Name |
|---|---------------|-------------------|------------|
| 1 | Bacterial Leaf Blight | Bacterial Leaf Blight | `bacterial_leaf_blight` |
| 2 | Brown Spot | Brown Spot | `brown_spot` |
| 3 | Rice Blast / Leaf Blast | Rice Blast / Leaf Blast | `rice_blast` / `leaf_blast` |
| 4 | Narrow Brown Leaf Spot | Narrow Brown Leaf Spot | `narrow_brown_leaf_spot` |
| 5 | False Smut | False Smut | `false_smut` |
| 6 | Dirty Seed | Dirty Seed | `dirty_seed` |
| 7 | Sheath Rot | Sheath Rot | `sheath_rot` |
| 8 | Stem Rot | Stem Rot | `stem_rot` |
| 9 | Red Stripe | Red Stripe | `red_stripe` |
| 10 | Healthy | Healthy | `healthy` |

---

## 🖼️ Rich Menu

The system includes a LINE Rich Menu with 3 buttons (vertical layout, 2500×1686 px):

| Button | Function |
|--------|----------|
| 🔝 **Rice Knowledge** | Opens the [Rice Knowledge Bank](https://rkb.ricethailand.go.th) website |
| 📸 **Select Photo** | Opens the device camera roll to send a photo for analysis |
| 🦠 **Rice Diseases** | Opens the rice disease information website |


---

## 🗄️ Database Schema

Uses **SQLite** (`data/rice_farmer.db`) with 3 tables:

### `users` — LINE User Profiles

| Column | Type | Description |
|--------|------|-------------|
| `line_user_id` | TEXT (UNIQUE) | LINE User ID |
| `display_name` | TEXT | LINE display name |
| `first_seen_at` | DATETIME | First interaction timestamp |
| `last_active_at` | DATETIME | Last activity timestamp |
| `total_analyses` | INTEGER | Total number of disease analyses |
| `total_messages` | INTEGER | Total number of messages sent |

### `analyses` — Rice Disease Analysis Results

| Column | Type | Description |
|--------|------|-------------|
| `line_user_id` | TEXT | LINE User ID |
| `disease` | TEXT | Detected disease name |
| `confidence` | REAL | Confidence score (0–1) |
| `severity` | TEXT | Severity level (Low / Medium / High) |
| `advice` | TEXT | Treatment advice from Gemini AI |
| `image_url` | TEXT | URL of the annotated image |

### `chat_history` — Conversation Logs

| Column | Type | Description |
|--------|------|-------------|
| `line_user_id` | TEXT | LINE User ID |
| `role` | TEXT | `user` or `bot` |
| `message` | TEXT | Message content |
| `created_at` | DATETIME | Timestamp |

---

## ⚙️ Internal Workflows

### Image Analysis Flow

```
1. User sends a rice leaf photo via LINE
2. Webhook receives the event → Replies "Analyzing..."
3. Downloads the image from LINE Content API
4. Sends the image to YOLO Server (/predict)
5. YOLO detects diseases → Returns predictions + annotated image
6. Calls Gemini AI to generate treatment advice (or uses fallback if API is unavailable)
7. Saves annotated image to public/results/
8. Stores results in SQLite + in-memory session
9. Pushes the annotated image + advice back to the user via LINE
```

### Chat Q&A Flow

```
1. User types a question about rice
2. Checks for greetings/farewells/thanks → Replies instantly (no AI call)
3. Attempts RAG Semantic Search first (ChromaDB)
4. If RAG is unavailable → Falls back to Keyword Search
5. Combines context (analysis results + chat history + disease knowledge) → Builds prompt
6. Calls Gemini AI (tries gemini-2.5-flash-lite first → falls back to gemini-2.5-flash)
7. Saves chat history + Replies to user via LINE
```

---

## 📝 License

This project is for educational purposes.

---

## 👨‍💻 Developers

Built by a student team to help Thai rice farmers 🌾
