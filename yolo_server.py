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

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODEL_PATH = "best.pt"

try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    print(f"📋 Classes: {model.names}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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

        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({
                "error": "Image too large",
                "message": "Image must be less than 10MB"
            }), 400

        if len(image_bytes) == 0:
            return jsonify({
                "error": "Empty image",
                "message": "Image file is empty"
            }), 400

        image_stream = io.BytesIO(image_bytes)
        try:
            # ใช้โค้ดโหลดรูปแบบเดียวกับใน Kaggle ของผู้ใช้เป๊ะๆ
            pil_img = Image.open(image_stream)
            print(f"📸 Original image size: {pil_img.size}")
        except Exception as e:
            return jsonify({
                "error": "Invalid image",
                "message": f"Cannot decode image: {str(e)}"
            }), 400

        # ทำนายด้วย YOLO แบบเดียวกับ Kaggle
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
            # ใช้ r.plot() แบบเดียวกับใน Kaggle
            annotated_img = results[0].plot() 
            _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            print(f"🖼️ Annotated image generated ({len(annotated_image_base64)} chars)")

        return jsonify({
            "success": True,
            "predictions": predictions,
            "total": len(predictions),
            "image_shape": [pil_img.height, pil_img.width, 3],
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
    app.run(host='0.0.0.0', port=5000, debug=False)