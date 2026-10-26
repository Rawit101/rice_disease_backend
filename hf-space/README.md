---
title: Rice Disease API
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# Rice Disease API — best.pt image detection

The default deployment loads the existing YOLO `best.pt` model without AI keys,
ChromaDB, generative-model SDKs or RAG initialization. Keep `ENABLE_RAG=false`.

- `POST /predict`: upload an image as multipart `image`.
- `GET /health`, `GET /classes`, `GET /`: detector information.

LINE sends `include_annotated_image=false` to skip annotation generation and transfer,
and `confidence_threshold` to match acceptance to inference. The default is
`YOLO_CONFIDENCE_THRESHOLD=0.25`. Other API consumers retain annotations by default.
No ONNX conversion, retraining or model replacement is required.

The image-only LINE bot replies to all text, follow-ups and unsupported inputs with
`กรุณาส่งรูปใบข้าวเพื่อตรวจโรค ระบบนี้รองรับเฉพาะรูปภาพ`; accepted detections produce
`🌾 ผลตรวจเบื้องต้น: {Thai disease name}`.

Legacy `/search` and `/query` endpoints remain only for separate standalone
consumers and return HTTP 503 by default. Explicit `ENABLE_RAG=true`, optional
`chromadb`/`google-genai` dependencies and that consumer's existing keys/data are
required to enable them. The Dockerfile supports `--build-arg ENABLE_RAG=true`
for such a separate deployment. Stale keys alone never enable RAG. LINE cannot
call these endpoints or regain conversational replies.

Sync this folder's app, Dockerfile and requirements with the existing Space mirror
while preserving its `best.pt`, knowledge and stored assets. See the repository
README for manual deployment, LINE OA menu/settings updates and verification.
No live publication is performed by changing these files.
