from ultralytics import YOLO
import cv2

# 1. โหลด Model
model = YOLO("best.pt")

# 2. ใส่พาธรูปภาพที่คุณต้องการทดสอบ
image_path = "test14.jpg" 

# 3. สั่งประมวลผล
# ปรับ conf=0.25 ตามที่คุณตั้งไว้ใน server
results = model.predict(source=image_path, conf=0.25, save=True)

# 4. แสดงผลลัพธ์ออกมาเป็นข้อความ
print("\n--- ผลการวิเคราะห์ ---")
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        conf = float(box.conf[0])
        print(f"พบโรค: {label} (Class ID: {class_id}) | ความมั่นใจ: {conf:.2%}")

print(f"\n✅ ตรวจสอบรูปผลลัพธ์ที่โฟลเดอร์: {results[0].save_dir}")