# ทำ label ร่างสำหรับรูปโรคข้าว

## คัดคุณภาพภาพก่อนทำ label — รอบ 3 พฤศจิกายน 2026

เจ้าของงานอนุญาตให้ไม่เลือกใช้ภาพที่ไม่เหมาะ จึงแยกการตรวจคุณภาพภาพออกจากความถูกต้องของ mask เดิม
ตรวจไฟล์/พิกเซลครบ 20,273 ไฟล์ พบภาพซ้ำแบบพิกเซลตรงกัน 2,736 สำเนา
เปิดดูรายภาพ 290 ภาพในรอบนี้: คัดเก็บ 91, ไม่เลือกด้านคุณภาพ 67,
พักเพราะอาการ/รายละเอียดไม่ชัด 75, ไม่เลือกมุมหรือฉบับใกล้เคียงซ้ำ 57 ภาพ

ผลนี้ **ยังไม่ใช่การเปิดดูรายภาพครบทั้ง 20,273 ไฟล์** ภาพที่ยังไม่เปิดดูคงสถานะรอตรวจ
ภาพที่มีคะแนนความคมสูงไม่ถูกตีความว่าเห็นอาการหรือยืนยันโรคแล้ว

- `temp/rice_labeling/rice_photos_selected.zip` มีเฉพาะภาพที่คัดเก็บ 91 ภาพ พร้อมรายชื่อและเว็บดูภาพ
- `temp/rice_labeling/rice_photo_quality_review/` ดูผลคัดทั้ง 290 ภาพพร้อมเหตุผล
- `temp/rice_labeling/rice_image_quality/photo_screening_all_files.csv` บันทึกสถานะทุกไฟล์
- ชื่อหมวดต้นทางยังไม่ยืนยัน และไม่พ่วง label เดิมในชุดภาพคัด ต้องระบุ class/วาดรอยอาการใหม่ก่อนเทรน

## ขอบเขตและการตรวจทบทวนล่าสุด — 1 พฤศจิกายน 2026

เจ้าของงานยืนยันเฉพาะ **9 โรคเดิม + Tungro รวม 10 คลาส** และเลือกให้
**ล้อมเฉพาะรอยหรือบริเวณอาการที่เห็น** ตาม [TEN_CLASS_POLICY.md](TEN_CLASS_POLICY.md)

ผลรอบเก่าด้านล่างเป็น pseudo-label และมีข้อผิดพลาดที่พบจากการเปิดดูภาพจริง
ห้ามตีความจำนวนไฟล์ label หรือการโหลดผ่าน YOLO ว่าเป็นจำนวนภาพที่ตรวจคุณภาพผ่านแล้ว

- ตรวจความเสี่ยงของไฟล์ในขอบเขตครบ 20,273 ภาพ มี mask เดิม 19,410 ภาพ และยังไม่มี mask 863 ภาพ
- เปิดเทียบ source/overlay 122 ภาพตาม class และความเสี่ยง ไม่ใช่การตรวจด้วยตาครบทุกภาพ
- พบ 7,990 ไฟล์ที่ class ของ label ขัดกับหมวดต้นทาง ซึ่งยังไม่ทราบว่าฝ่ายใดถูก
- หมวดนอกขอบเขต 11,907 ไฟล์ถูกแยกจากชุดใหม่ รวม Healthy; ไม่เพิ่มคลาสที่ 11
- สี/SAM และความเห็นตรงกันของโมเดลสองเวอร์ชันไม่ใช่การยืนยันโรค
- `temp/rice_labeling/rice_friend_review_10class.zip` เป็นชุดตรวจใหม่ แยก `draft_labels` และ `held_labels`
- ชุดใหม่ยังไม่ใช่ชุดพร้อมเทรน ไม่มี train split ที่อ้างว่าตรวจผ่านแล้ว

รายงานอยู่ใน `temp/rice_labeling/rice_10class_audit/` และ `rice_10class_review/`
มีข้อเสนอแก้กรณีดอกกระถินที่ตกหล่นจาก 1 เป็น 3 ก้อน โดยคง class เดิมและเก็บต้นฉบับไว้
ข้อมูลจากรอบตรวจเป็น risk signals ไม่ใช่อัตราความแม่นยำหรือการรับรองชื่อโรคทั้งชุด

## เทียบโมเดลและส่งออกอัตโนมัติรอบล่าสุด

`temp/rice_labeling/rice_full_auto.zip` มีรูปครบ 2,817 ไฟล์และผลเปรียบเทียบโมเดล
ชุดฝึก v8/v6 ทุกรูป รวม healthy และโรคใหม่ ทั้งสองโมเดลยังมีเพียง 9 คลาสเดิม
ได้ annotation 2,509 ไฟล์ (รวมรูปซ้ำและภาพพื้นหลัง 140 ไฟล์)
ชุด `data.yaml` ใช้ train 1,950 ภาพ และ validation ที่ AI ตรวจแล้ว 34 ภาพ
YOLO สแกนรูป/ขอบเขตผ่าน ไม่มี corrupt และสร้าง batch ได้จริง

**เป็น pseudo-label อัตโนมัติ ไม่ใช่ ground truth ที่ยืนยันแล้วทั้งชุด**
ยังมี 308 ไฟล์ไม่มีขอบเขตโรค: 163 ไฟล์ที่โมเดลยังหาเป้าหมายไม่ได้ และ 145 ไฟล์โรคใหม่
ผลที่ทราบจากการดูภาพแล้วว่ามีปัญหาไม่อยู่ใน train ปกติ
`data_all_predictions.yaml` เก็บผลทำนายกว้างกว่าเพื่อให้ตรวจแก้ ไม่ใช่ชุดที่รับรองคุณภาพ
เปิด `compare.html` ผ่านเซิร์ฟเวอร์ตาม README ในแพ็ก เพื่อเทียบภาพและแต่ละโมเดล
มี `image_status.csv`, `report.json` และบันทึกตรวจภาพที่ยังตกหล่นรายภาพ
ยังต้องรวมกับชุดเทรนเดิมและใช้ val/test เดิมในการประเมินทั้ง 9 คลาส

```bash
venv/bin/python scripts/validate_full_rice.py \
  --root temp/rice_labeling/rice_full_auto
```

## แพ็กตรวจเข้มงวดรอบก่อน

`temp/rice_labeling/rice_training_handoff.zip` เป็นแพ็กที่มีรูปครบ 2,817 ไฟล์
พร้อม `data.yaml`, train/val lists, สำเนา best.pt เดิม และสคริปต์รวมกับชุดเทรนเดิม
อ่าน `SUMMARY_TH.md` และ `README_TH.md` ภายในแพ็กก่อนใช้

**งาน annotation ทั้งชุดยังไม่เสร็จ**: ผ่านเกณฑ์ 167 ภาพไม่ซ้ำ
(ภาพโรค 48 + ภาพพื้นหลังที่ AI ดูเพิ่ม 119) แบ่ง train 133 / val 34
มี label ให้สำเนาซ้ำอีก 8 ไฟล์ แต่ไม่ใส่ซ้ำใน train/val
อีก 2,642 ไฟล์ยังใช้เทรนไม่ได้ และอยู่ใน `excluded.csv` พร้อมเหตุผล
การตรวจด้วยโมเดลสำรองอีกตัวไม่เพิ่มจำนวนภาพที่ผ่านเกณฑ์

ภาพโรคที่ผ่านเกณฑ์มีเพียง Rice_Blast, False_Smut และ Stem_Rot
ต้องใช้เป็นข้อมูลเพิ่มร่วมกับชุดเดิม ไม่ใช่ใช้แทนข้อมูลเดิมทั้งหมดทั้ง 9 คลาส
ยังไม่มี label โรคใหม่ที่พร้อมใช้งาน และ archive.zip จาก Drive ยังดาวน์โหลดไม่เสร็จ
การตรวจภาพโดย AI ไม่ใช่การรับรองชื่อโรคโดยผู้เชี่ยวชาญ

```bash
venv/bin/python scripts/handoff_validate.py \
  --root temp/rice_labeling/rice_training_handoff --load-images
```

เครื่องมืออยู่ที่ `scripts/auto_label_rice.py` ใช้โมเดล Segmentation เดิมใน `best.pt`
เพื่อสร้าง polygon ร่าง ไม่มีการเทรนหรือแก้โมเดลเดิม และไม่มีการตัดรูปทิ้ง

ชุดแรกใช้รูป 2,817 ไฟล์จาก 12 โฟลเดอร์ที่ผู้ใช้ระบุใน Downloads
ผลอยู่ที่ `temp/rice_labeling/local_all/` ซึ่งไม่ถูกเพิ่มเข้า Git

## เปิดผล

เปิด `temp/rice_labeling/local_all/review.html` ในเบราว์เซอร์ ดูภาพพร้อมขอบเขต
และกรองตามชื่อกลุ่มหรือสถานะได้ สถานะทุกภาพยังเป็น **ไม่ผ่านการยืนยัน**

- `labels_draft/`: polygon ที่โมเดลโรคเดิมทาย พร้อมหมายเลขคลาสเดิม 0–8
- `proposals/`: ขอบเขตวัตถุทั่วไปจาก FastSAM ไม่มีชื่อโรค ไม่ใช่ label พร้อมเทรน
- `images/`: รูปที่จับคู่กับผลลัพธ์ หมุนตาม EXIF และแปลง RGB JPEG แล้ว
- `manifest.csv`: สถานะและที่มาทุกรูป รวมรูปที่ทำ label ไม่ได้
- `summary.json`, `folder_summary.json`: จำนวนผลลัพธ์แยกสถานะและกลุ่ม
- `duplicate_groups.json`: ไฟล์ที่เนื้อหาเหมือนกัน เก็บไว้ทุกรูปแต่บันทึกกลุ่มซ้ำให้ผู้เทรน
- `classes.yaml`: รายชื่อคลาส หมายเลขใหม่เป็นข้อเสนอให้ผู้เทรนตรวจสอบ

การไม่พบโรคไม่ได้แปลว่ารูปนั้นปกติ ไม่มีการสร้างไฟล์ label ว่างเพื่ออ้างว่าเป็น background
รูปที่ชื่อโรคในโฟลเดอร์ไม่ตรงกับผลทำนายมีธงเตือน โดยไม่เปลี่ยนคลาสของผลทำนายให้เอง
ภาพตัวอย่างการเทรนเดิมล้อมบริเวณอาการ ขอบเขตทั้งใบจาก FastSAM จึงต้องตรวจแก้ก่อนใช้

## ทำต่อหลังถูกขัดจังหวะ

```bash
venv/bin/python scripts/auto_label_rice.py run \
  --output temp/rice_labeling/local_all --model best.pt --device auto
```

โปรแกรมอ่าน `progress.sqlite` และข้ามรายการที่ทำเสร็จแล้ว
ถ้าต้องการเพิ่มข้อเสนอวัตถุให้รูปที่ยังไม่มี polygon โรค:

```bash
venv/bin/python scripts/auto_label_rice.py run \
  --output temp/rice_labeling/local_all --model best.pt --device auto \
  --proposal-model temp/rice_labeling/models/FastSAM-s.pt --retry-unlabeled
```

## ชุดเพิ่มเติม 18 กลุ่ม

เมื่อดาวน์โหลด `archive.zip` จาก Drive และแตกไฟล์แล้ว ให้ระบุโฟลเดอร์ `Rice Disease` จริง
แทน `/path/to/Rice Disease` (อย่ารันคำสั่งพร้อม placeholder):

```bash
venv/bin/python scripts/auto_label_rice.py run \
  --source '/path/to/Rice Disease' --output temp/rice_labeling/drive_all \
  --model best.pt --device auto \
  --proposal-model temp/rice_labeling/models/FastSAM-s.pt
```

ถ้าพาธมีอยู่แล้วใน output เดิมจะไม่เพิ่มซ้ำ รูปซ้ำคนละพาธยังอยู่ครบ
ผลทำนายเหมือนกันจะใช้ cache ลดการประมวลผลซ้ำ คะแนนอยู่ในฐานข้อมูลและ cache แยกจาก
ไฟล์ YOLO ซึ่งมีเฉพาะ class_id กับพิกัด polygon แบบ normalized ไม่มีคะแนนต่อท้าย

## ส่งให้ผู้เทรน

```bash
venv/bin/python scripts/auto_label_rice.py package \
  --output temp/rice_labeling/local_all --include-images
```

สร้าง `temp/rice_labeling/local_all_with_images.zip` มีรูปและผลร่างจับคู่กัน
ถ้าไม่ใส่ `--include-images` จะได้ ZIP เฉพาะ label ข้อเสนอ และรายงาน
ต้องระบุให้ผู้เทรนทราบว่าทั้งหมดเป็น AI drafts; ยังไม่มี train/val/test หรือ ground truth
ผู้เทรนควรแยกรูปซ้ำและรูปดัดแปลงจากต้นฉบับเดียวกันให้อยู่ชุดเดียวกันก่อนประเมินโมเดล

## ทดสอบตัวส่งออก

```bash
venv/bin/python -m unittest discover -s tests -p test_auto_label_rice.py -v
```
