"""
สคริปต์รวม 3 รูปเป็น Rich Menu ขนาด 2500x1686 (3 แถวแนวตั้ง)
วิธีใช้: python combine_richmenu.py

ต้องวางรูป 3 ไฟล์ไว้ในโฟลเดอร์เดียวกัน:
  - menu1.jpg หรือ .png  → องค์ความรู้เรื่องข้าว (บน)
  - menu2.jpg หรือ .png  → เลือกรูปภาพ (กลาง)
  - menu3.jpg หรือ .png  → โรคข้าว (ล่าง)

ผลลัพธ์: richmenu.jpg (2500x1686, < 1MB)
"""

from PIL import Image
import os
import sys

# ขนาด Rich Menu เต็มจอ
MENU_WIDTH = 2500
MENU_HEIGHT = 1686

# แบ่งเป็น 3 แถว
ROW_HEIGHT = MENU_HEIGHT // 3  # 562

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'richmenu.jpg')

def find_file(prefix):
    """หาไฟล์ที่ชื่อขึ้นต้นด้วย prefix และลงท้ายด้วย png/jpg/jpeg"""
    for ext in ['.jpg', '.png', '.jpeg']:
        path = os.path.join(BASE_DIR, f"{prefix}{ext}")
        if os.path.exists(path):
            return path
    return None

def resize_and_crop(img, target_width, target_height):
    """ปรับขนาดรูปให้พอดี (cover)"""
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # รูปกว้างกว่า → resize ตามความสูง แล้ว crop ซ้ายขวา
        new_height = target_height
        new_width = int(target_height * img_ratio)
    else:
        # รูปสูงกว่า → resize ตามความกว้าง แล้ว crop บนล่าง
        new_width = target_width
        new_height = int(target_width / img_ratio)

    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return img_resized.crop((left, top, right, bottom))

def main():
    print("🖼️  สร้าง Rich Menu Image (2500x1686 - 3 แถวแนวตั้ง)")
    print("=" * 45)

    menu1_path = find_file('menu1')
    menu2_path = find_file('menu2')
    menu3_path = find_file('menu3')

    missing = []
    if not menu1_path: missing.append("menu1.jpg")
    if not menu2_path: missing.append("menu2.jpg")
    if not menu3_path: missing.append("menu3.jpg")

    if missing:
        print("❌ ไม่พบไฟล์รูปต่อไปนี้:")
        for f in missing:
            print(f"   - {f}")
        print(f"\n📂 กรุณานำรูปที่เซฟมาวางไว้ที่: {BASE_DIR}")
        sys.exit(1)

    print("📂 โหลดรูป...")
    img1 = Image.open(menu1_path).convert('RGB')
    img2 = Image.open(menu2_path).convert('RGB')
    img3 = Image.open(menu3_path).convert('RGB')

    print("\n🔧 ปรับขนาด...")
    part1 = resize_and_crop(img1, MENU_WIDTH, ROW_HEIGHT)
    part2 = resize_and_crop(img2, MENU_WIDTH, ROW_HEIGHT)
    part3 = resize_and_crop(img3, MENU_WIDTH, ROW_HEIGHT)

    print("\n🎨 รวมรูป...")
    canvas = Image.new('RGB', (MENU_WIDTH, MENU_HEIGHT), color=(255, 255, 255))
    
    canvas.paste(part1, (0, 0))
    canvas.paste(part2, (0, ROW_HEIGHT))
    canvas.paste(part3, (0, ROW_HEIGHT * 2))

    print("\n💾 บันทึก...")
    quality = 95
    while quality >= 60:
        canvas.save(OUTPUT_FILE, 'JPEG', quality=quality, optimize=True)
        size_bytes = os.path.getsize(OUTPUT_FILE)
        
        if size_bytes <= 1024 * 1024:
            break
            
        print(f"   ⚠️ ไฟล์ > 1MB, ลดคุณภาพ... (quality={quality-5})")
        quality -= 5

    size_kb = size_bytes / 1024
    
    print(f"\n{'=' * 45}")
    print(f"✅ สร้าง Rich Menu สำเร็จ!")
    print(f"   📄 ไฟล์: {OUTPUT_FILE}")
    print(f"   📐 ขนาด: {MENU_WIDTH}x{MENU_HEIGHT} px")
    print(f"   💽 ขนาดไฟล์: {size_kb:.1f} KB")
    print(f"\n📌 ขั้นตอนถัดไป: รันคำสั่ง node setup-richmenu.js")

if __name__ == '__main__':
    main()
