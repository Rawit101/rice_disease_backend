---
title: Rice Disease API
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# Rice Disease API

YOLO + RAG API server for rice disease detection and knowledge search.

## Endpoints

- `POST /predict` — YOLO rice disease detection (upload image)
- `POST /search` — RAG semantic search for disease info
- `POST /query` — RAG search + AI-generated answer
- `GET /health` — Health check
