"""Build an offline, portable review website with original images and YOLO labels."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import shutil
import zipfile

SCRIPTS = Path(__file__).resolve().parent
BASE = SCRIPTS.parent / 'temp/rice_labeling'


def build():
    source = BASE / 'rice_named_labels'
    out = BASE / 'rice_friend_review'
    for folder in ['images', 'labels', 'annotations']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / 'manifest.json').read_text())
    classes = json.loads((source / 'class_names_th.json').read_text())
    portable, human, checked = [], [], []
    fingerprint = hashlib.sha256()
    for index, r in enumerate(rows, 1):
        image = (source / r['image']).resolve()
        label = (source / r['label']).resolve()
        data = json.loads((source / r['annotation']).read_text())
        raw = label.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == data['label_sha256'], r['id']
        assert sorted(set(o['class_id'] for o in data['objects'])) == r['class_ids']
        fingerprint.update((r['id'] + ':' + sha + '\n').encode())
        image_name = 'images/' + r['id'] + image.suffix
        label_name = 'labels/' + r['id'] + '.txt'
        annotation_name = 'annotations/' + r['id'] + '.js'
        shutil.copyfile(image, out / image_name)
        shutil.copyfile(label, out / label_name)
        # Script loading also works when index.html is opened directly with file://.
        # No fetch(), cloud service, Python, or credentials are needed for the viewer.
        annotation = dict(objects=[dict(class_id=o['class_id'], polygon=o['polygon']) for o in data['objects']])
        (out / annotation_name).write_text('window.RiceReviewAnnotations[' + json.dumps(r['id']) + ']='
                                          + json.dumps(annotation, separators=(',', ':')) + ';\n')
        portable.append(dict(r, image=image_name, label=label_name, annotation=annotation_name))
        human.append(dict(id=r['id'], filename=r['filename'], dataset=r['dataset'],
                          class_ids=' | '.join(map(str, r['class_ids'])),
                          class_names_en=' | '.join(classes[i]['name_en'] for i in r['class_ids']),
                          class_names_th=' | '.join(classes[i]['name_th'] for i in r['class_ids']),
                          object_count=r['object_count'], quality=r['quality'], label_sha256=sha))
        checked.append((label, out / label_name, sha))
        if index % 2000 == 0:
            print(json.dumps(dict(stage='copying_review_images', completed=index, total=len(rows))), flush=True)
    catalog = dict(manifest=portable, classes=classes, fingerprint=fingerprint.hexdigest())
    (out / 'catalog.js').write_text('window.RiceReviewCatalog=' + json.dumps(catalog, ensure_ascii=False, separators=(',', ':')) + ';\n')
    shutil.copyfile(SCRIPTS / 'rice_friend_review.html', out / 'index.html')
    shutil.copyfile(SCRIPTS / 'rice_class_names.js', out / 'rice_class_names.js')
    shutil.copyfile(source / 'classes.csv', out / 'classes.csv')
    with (out / 'image_classes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(human[0]))
        writer.writeheader()
        writer.writerows(human)
    (out / 'README_TH.txt').write_text('''เว็บตรวจ label โรคข้าว สำหรับส่งให้เพื่อน

1. ดาวน์โหลด ZIP แล้วแตกไฟล์ทั้งหมดก่อน อย่าเปิด index.html จากใน ZIP
2. เปิดโฟลเดอร์ rice_friend_review แล้วดับเบิลคลิก index.html ด้วย Chrome/Edge/Firefox/Safari
   เว็บ รูป และขอบเขตอยู่ในชุดนี้ครบ ไม่ต้องติดตั้งโปรแกรมหรือใช้อินเทอร์เน็ต
3. เลือก class/ชุดข้อมูล ค้นชื่อไฟล์ หรือกดลูกศรเพื่อดูภาพ
4. เลือกผลตรวจ: เห็นว่าถูกต้อง / ต้องแก้ไข / ยังไม่แน่ใจ และพิมพ์หมายเหตุ
5. กด ส่งออกผลตรวจ CSV และ สำรองผลตรวจ JSON แล้วส่งไฟล์ทั้งสองคืนเจ้าของงาน
   ผลตรวจเก็บในเบราว์เซอร์ของเครื่องนั้น ไม่ได้ส่งกลับเจ้าของงานอัตโนมัติ
   สำรอง JSON ก่อนล้างข้อมูลเบราว์เซอร์ เปลี่ยนเบราว์เซอร์ หรือย้ายเครื่อง
   นำเข้า JSON เพื่อทำต่อได้ โดยผลตรวจที่มีเวลาล่าสุดของแต่ละภาพจะถูกเก็บไว้

มีรูปที่มีไฟล์ label 19,550 รูป: รูปที่มีขอบเขต 19,410 รูป + label ว่าง 140 รูป
เป็นไฟล์จากทุกชุดที่ label ไว้ รวมรูปซ้ำ ฉบับร่าง และภาพที่พักไว้ ไม่ใช่จำนวนพร้อมเทรน
ไม่มีภาพที่ยังทำ label ไม่สำเร็จอยู่ในชุดนี้
ชื่อโรคอ่านจาก class ของ label ไม่ใช่ชื่อโฟลเดอร์ และยังไม่มีผู้เชี่ยวชาญยืนยัน
ผลตรวจและหมายเหตุไม่แก้รูปหรือไฟล์ YOLO อัตโนมัติ
images/ เป็นสำเนารูปขนาดเดิม labels/ เป็นไฟล์ YOLO เดิม พร้อมเลข class 0–9
classes.csv และ image_classes.csv แสดงชื่อโรคไทย–อังกฤษกำกับเลข class
ไม่มีไฟล์โมเดลหรือข้อมูลเข้าสู่ระบบในชุดนี้

หากเบราว์เซอร์บล็อกการเปิดไฟล์ในเครื่อง และมี Python อยู่แล้ว:
เปิด Terminal/Command Prompt ในโฟลเดอร์นี้ แล้วรัน python serve.py
(Windows อาจใช้ py serve.py / macOS ใช้ python3 serve.py)
จากนั้นเปิด http://127.0.0.1:8774/index.html ในเครื่องที่รันคำสั่ง
ลิงก์ 127.0.0.1 ใช้ได้เฉพาะเครื่องนั้น เวลาส่งให้คนอื่นให้ส่ง ZIP ทั้งชุด
''', encoding='utf-8-sig')
    (out / 'serve.py').write_text('''"""Optional local-only server for the offline review website."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    print('Open http://127.0.0.1:8774/index.html (Ctrl+C to stop)', flush=True)
    ThreadingHTTPServer(('127.0.0.1',8774), partial(SimpleHTTPRequestHandler,directory=str(root))).serve_forever()
''')
    assert len({r['id'] for r in portable}) == len(portable)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == sha and hashlib.sha256(q.read_bytes()).hexdigest() == sha
               for p, q, sha in checked)
    report = dict(source_files=len(rows), positive_files=sum(bool(r['class_ids']) for r in rows),
                  empty_label_files=sum(not r['class_ids'] for r in rows), class_count=len(classes),
                  dataset_counts=dict(Counter(r['dataset'] for r in rows)),
                  original_label_bytes_preserved=True, review_results_modify_labels=False,
                  original_image_bytes=sum((out / r['image']).stat().st_size for r in portable),
                  fingerprint=catalog['fingerprint'])
    (out / 'package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    target = BASE / 'rice_friend_review.zip'
    files = sorted(p for p in out.rglob('*') if p.is_file())
    with zipfile.ZipFile(target, 'w', allowZip64=True) as archive:
        for index, p in enumerate(files, 1):
            archive.write(p, 'rice_friend_review/' + str(p.relative_to(out)),
                          compress_type=zipfile.ZIP_STORED if p.suffix.lower() in ['.jpg', '.jpeg', '.png'] else zipfile.ZIP_DEFLATED)
            if index % 15000 == 0:
                print(json.dumps(dict(stage='packing_review_website', completed=index, total=len(files))), flush=True)
    print(json.dumps(dict(stage='checking_review_zip_crc', bytes=target.stat().st_size)), flush=True)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert sum(n.startswith('rice_friend_review/images/') for n in names) == len(rows)
        assert sum(n.startswith('rice_friend_review/labels/') for n in names) == len(rows)
        assert sum(n.startswith('rice_friend_review/annotations/') for n in names) == len(rows)
        assert 'rice_friend_review/index.html' in names
    print(json.dumps(dict(stage='portable_review_zip_passed', zip=str(target), bytes=target.stat().st_size, **report)), flush=True)


if __name__ == '__main__':
    build()
