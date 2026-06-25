from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# จำกัดขนาดไฟล์ upload สูงสุด 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# โหลด YOLO model
MODEL_PATH = "best.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"📋 Classes: {model.names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Disease names — ใช้ model.names โดยตรง (ดู /classes endpoint)

# สีสำหรับแต่ละโรค (BGR format)
DISEASE_COLORS = {
    'brown_spot': (0, 100, 255),             # ส้ม
    'leaf_blast': (0, 0, 255),               # แดง
    'bacterial_leaf_blight': (0, 255, 255),  # เหลือง
    'narrow_brown_leaf_spot': (255, 0, 255), # ม่วง
    'healthy': (0, 255, 0),                   # เขียว
    'Rice_Blast': (0, 0, 255),               # แดง
    'Rice_blast': (0, 0, 255)                # แดง
}

MAX_IMAGE_SIZE = 640


def resize_image(img, max_size=MAX_IMAGE_SIZE):
    """Resize image ให้ไม่เกิน max_size เพื่อประหยัด memory และเพิ่มความเร็ว"""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"📐 Resized image: {w}x{h} -> {new_w}x{new_h}")
    return resized


def draw_detections(img, predictions):
    """วาด bounding box และ label บนรูปภาพ"""
    annotated = img.copy()
    h, w = annotated.shape[:2]

    for pred in predictions:
        bbox = pred['bbox']
        class_name = pred['class']
        confidence = pred['confidence']

        # พิกัด bounding box
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # สีตามโรค
        color = DISEASE_COLORS.get(class_name, (0, 0, 255))

        # ความหนาของเส้นตามขนาดรูป
        thickness = max(2, int(min(h, w) / 200))

        # วาด bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label text
        label = f"{class_name} {confidence:.0%}"

        # ขนาด font ตามขนาดรูป
        font_scale = max(0.5, min(h, w) / 800)
        font_thickness = max(1, int(min(h, w) / 400))

        # คำนวณขนาด label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # วาด label background
        label_y1 = max(y1 - label_h - baseline - 10, 0)
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + label_w + 10, y1),
            color, -1
        )

        # วาด label text (สีดำ)
        cv2.putText(
            annotated, label,
            (x1 + 5, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), font_thickness, cv2.LINE_AA
        )

    return annotated


@app.route('/')
def home():
    return jsonify({
        "status": "YOLO API is running",
        "model": MODEL_PATH,
        "model_loaded": model is not None,
        "classes": model.names if model else {}
    })


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "message": f"Please check if {MODEL_PATH} exists"
        }), 500

    if 'image' not in request.files:
        return jsonify({
            "error": "No image file provided",
            "message": "Please send image with key 'image'"
        }), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()

        # ตรวจสอบขนาดไฟล์ (ไม่เกิน 10MB)
        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({
                "error": "Image too large",
                "message": "Image must be less than 10MB"
            }), 400

        # ตรวจสอบว่าไฟล์ไม่ว่าง
        if len(image_bytes) == 0:
            return jsonify({
                "error": "Empty image",
                "message": "Image file is empty"
            }), 400

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({
                "error": "Invalid image",
                "message": "Cannot decode image"
            }), 400

        print(f"📸 Original image: {img.shape}")

        # Resize image ก่อนส่ง YOLO เพื่อประหยัด memory
        img = resize_image(img)

        # ทำนายด้วย YOLO
        results = model(img, conf=0.25)

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

        # วาด bounding box บนรูปแล้ว encode เป็น base64
        annotated_image_base64 = None
        if len(predictions) > 0:
            annotated_img = draw_detections(img, predictions)
            _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"🖼️ Annotated image generated ({len(annotated_image_base64)} chars)")

        return jsonify({
            "success": True,
            "predictions": predictions,
            "total": len(predictions),
            "image_shape": list(img.shape),
            "annotated_image": annotated_image_base64
        })

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@app.route('/classes')
def get_classes():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({
        "classes": model.names,
        "total": len(model.names)
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Starting YOLO API Server")
    print("=" * 50)
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"✅ Model loaded: {model is not None}")
    if model:
        print(f"📋 Classes: {model.names}")
    print("=" * 50)

    # Production: debug=False เพื่อความเร็วและความปลอดภัย
    app.run(host='0.0.0.0', port=5000, debug=False)