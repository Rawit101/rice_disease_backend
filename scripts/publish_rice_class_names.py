"""Add human-readable names to existing annotations without modifying training data."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import shutil
import zipfile

SCRIPTS = Path(__file__).resolve().parent
BASE = SCRIPTS.parent / 'temp/rice_labeling'
NAMES = ['Bacterial_Blight', 'Brown_Spot', 'Rice_Blast', 'Narrow_Brown_Spot',
         'False_Smut', 'Dirty_Seed', 'Sheath_Rot', 'Stem_Rot', 'Red_Stripe', 'Tungro']
THAI = ['โรคขอบใบแห้ง', 'โรคใบจุดสีน้ำตาล', 'โรคไหม้', 'โรคใบขีดสีน้ำตาล',
        'โรคดอกกระถิน', 'โรคเมล็ดด่าง', 'โรคกาบใบเน่า', 'โรคลำต้นเน่า',
        'โรคใบแถบแดง', 'โรคใบสีส้ม']
CLASSES = [dict(class_id=i, name_en=en, name_th=THAI[i]) for i, en in enumerate(NAMES)]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_label(text, names):
    """Resolve IDs using this export's class table, never source_class_id."""
    objects = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        class_id = int(values[0])
        if not 0 <= class_id < len(names):
            raise ValueError('Unknown annotation class on line %d' % index)
        coords = list(map(float, values[1:]))
        if len(coords) < 6 or len(coords) % 2 or not all(0 <= v <= 1 for v in coords):
            raise ValueError('Invalid polygon on line %d' % index)
        en = names[class_id]
        objects.append(dict(class_id=class_id, name_en=en, name_th=THAI[NAMES.index(en)],
                            polygon=list(map(list, zip(coords[::2], coords[1::2])))))
    return objects


def update_viewers():
    helper = (SCRIPTS / 'rice_class_names.js').read_text()
    for en, th in zip(NAMES, THAI):
        assert re.search(r'\b' + en + r":\s*'" + th + "'", helper), en
    targets = [
        ('rice_tungro_review.html', 'rice_tungro_labeled_v2/review.html'),
        ('rice_corrections_review.html', 'rice_archive_corrected/corrections.html'),
        ('rice_model_compare.html', 'rice_full_auto/compare.html'),
        ('rice_archive_compare.html', 'rice_archive_auto/compare.html'),
        ('rice_remaining_review.html', 'remaining_recheck/review.html'),
    ]
    start, end = '<!-- BEGIN CLASS NAME RENDERER -->', '<!-- END CLASS NAME RENDERER -->'
    for filename, destination in targets:
        path = SCRIPTS / filename
        text = path.read_text()
        text = re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\n?', '', text, flags=re.S)
        if filename in ('rice_tungro_review.html', 'rice_corrections_review.html'):
            pattern = r'function paint\(id,img,objects\)\{.*?\n(?=(?:async function render|function info))'
            text, count = re.subn(pattern,
                "function paint(id,img,objects){RiceClassNames.draw($(id),img,objects,names)}\n", text, flags=re.S)
            assert count == 1
        elif filename in ('rice_model_compare.html', 'rice_archive_compare.html'):
            text, count = re.subn(r'function canvas\(id,img,objects\)\{.*?\n(?=function paint)',
                "function canvas(id,img,objects){RiceClassNames.draw($(id),img,$('show').checked?objects:[],names)}\n", text, flags=re.S)
            assert count == 1
        else:
            # Keep unaccepted crop proposals as proposals, merely give their class a readable name.
            if "RiceClassNames.draw($('overlay'),img,data.objects,names);}" not in text:
                text, count = re.subn(r"const c=\$\('overlay'\);.*?\n(?=for\(const id|function chip)",
                    "RiceClassNames.draw($('overlay'),img,data.objects,names);}\n", text, flags=re.S)
                assert count == 1
        if filename == 'rice_tungro_review.html':
            text = text.replace('let all=[],rows=[],pos=0,seq=0;', 'let all=[],rows=[],pos=0,seq=0,names=[];')
            text = text.replace('all=m;$(\'summary\')', "all=m;names=r.class_names;$('summary')")
            text = text.replace('`${data.objects.length} บริเวณ · class 9: Tungro`',
                                '`${data.objects.length} บริเวณ · ${RiceClassNames.summary(data.objects,names)}`')
        if filename == 'rice_corrections_review.html':
            text = text.replace("[...new Set(objects.map(o=>names[o.class_id]||o.class_id))].join(', ')",
                                'RiceClassNames.summary(objects,names)')
        # Inline the shared renderer so existing export scripts still produce standalone viewers.
        text = text.replace('<script>\n', start + '\n<script>\n' + helper + '\n</script>\n' + end + '\n<script>\n', 1)
        path.write_text(text)
        shutil.copyfile(path, BASE / destination)


def publish():
    out = BASE / 'rice_named_labels'
    (out / 'annotations').mkdir(parents=True, exist_ok=True)
    all_rows, csv_rows, checks, dataset_counts = [], [], [], Counter()
    fields = ['id', 'dataset', 'filename', 'source', 'image', 'label', 'class_ids',
              'class_names_en', 'class_names_th', 'object_count', 'quality', 'split',
              'expert_diagnosis_confirmed', 'label_sha256']
    inputs = ['rice_full_auto', 'rice_archive_corrected', 'rice_tungro_labeled_v2', 'remaining_recheck']
    for dataset in inputs:
        root = BASE / dataset
        if dataset == 'remaining_recheck':
            root = root / 'reviewed_additions'
            rows = json.loads((root / 'manifest.json').read_text())
            names = NAMES[:9]
            rows = [dict(r, image='images/' + r['id'] + '.jpg', label='labels/' + r['id'] + '.txt',
                         quality='ai_visually_reviewed_manual_addition', default_split='held_for_merge') for r in rows]
        else:
            names = json.loads((root / 'report.json').read_text())['class_names']
            rows = json.loads((root / 'manifest.json').read_text())
        assert names == NAMES[:len(names)], 'Unexpected class table: ' + dataset
        per_csv = []
        for r in rows:
            if not r.get('label'):
                continue
            image, label = root / r['image'], root / r['label']
            assert image.is_file() and label.is_file(), r['id']
            raw = label.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            objects = parse_label(raw.decode('utf-8'), names)
            ids = sorted(set(o['class_id'] for o in objects))
            annotation = 'annotations/' + r['id'] + '.json'
            # The polygon coordinates are copied from the actual YOLO file.
            write_json(out / annotation, dict(id=r['id'], dataset=dataset, source=r['source'],
                label_source=str(label.relative_to(BASE)), label_sha256=sha,
                expert_diagnosis_confirmed=False, quality=r['quality'], objects=objects))
            row = dict(id=r['id'], dataset=dataset, filename=Path(r['source']).name,
                       image='../' + str(image.relative_to(BASE)), label='../' + str(label.relative_to(BASE)),
                       annotation=annotation, class_ids=ids, object_count=len(objects),
                       quality=r['quality'], split=r.get('default_split'), exact_duplicate=r.get('exact_duplicate', False))
            all_rows.append(row)
            human = dict(id=r['id'], dataset=dataset, filename=row['filename'], source=r['source'],
                         image=str(image.relative_to(BASE)), label=str(label.relative_to(BASE)),
                         class_ids=' | '.join(map(str, ids)), class_names_en=' | '.join(names[i] for i in ids),
                         class_names_th=' | '.join(THAI[i] for i in ids), object_count=len(objects),
                         quality=r['quality'], split=row['split'], expert_diagnosis_confirmed=False, label_sha256=sha)
            csv_rows.append(human)
            per_csv.append(human)
            checks.append((label, sha))
            dataset_counts[dataset] += 1
        write_json(root / 'class_names_th.json', CLASSES[:len(names)])
        write_csv(root / 'classes.csv', ['class_id', 'name_en', 'name_th'], CLASSES[:len(names)])
        write_csv(root / 'image_classes.csv', fields, per_csv)
        print(json.dumps(dict(stage='names_added', dataset=dataset, labeled_files=len(per_csv))), flush=True)
    assert len({r['id'] for r in all_rows}) == len(all_rows), 'Duplicate source IDs need precedence'
    write_json(out / 'manifest.json', all_rows)
    write_json(out / 'class_names_th.json', CLASSES)
    write_csv(out / 'classes.csv', ['class_id', 'name_en', 'name_th'], CLASSES)
    write_csv(out / 'image_classes.csv', fields, csv_rows)
    report = dict(labeled_source_files=len(all_rows), positive_source_files=sum(bool(r['class_ids']) for r in all_rows),
                  empty_background_labels=sum(not r['class_ids'] for r in all_rows), dataset_counts=dict(dataset_counts),
                  annotation_objects=sum(r['object_count'] for r in all_rows), class_names=NAMES,
                  all_label_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==sha for p,sha in checks),
                  expert_diagnoses_confirmed=False, training_executed=False,
                  count_note='Source files including duplicates and held/draft annotations; not training-ready image counts.')
    assert report['all_label_hashes_unchanged']
    write_json(out / 'report.json', report)
    shutil.copyfile(SCRIPTS / 'rice_named_labels_review.html', out / 'review.html')
    shutil.copyfile(SCRIPTS / 'rice_class_names.js', out / 'rice_class_names.js')
    (out / 'README_TH.md').write_text('''# ชื่อ class ของ annotation ทุกชุด

image_classes.csv จับคู่ชื่อไฟล์กับเลข class ชื่อไทย ชื่ออังกฤษ และสถานะเดิม
classes.csv / class_names_th.json ระบุ class 0–9 โดย Tungro = 9
annotations/*.json มีชื่อโรคและ polygon ทุกขอบเขต อ่านจากไฟล์ YOLO ที่ใช้งานจริง
label_sha256 ใช้ตรวจว่าไฟล์ YOLO ต้นทางยังตรงกับฉบับที่ระบุชื่อ

ชุดนี้เป็นข้อมูลชื่อประกอบ annotation ไม่รวมรูปต้นฉบับ ไม่ใช่ชุดเทรนใหม่
จำนวนรวมเป็นไฟล์ภาพที่มี label รวมรูปซ้ำ/ฉบับร่าง/ภาพที่พักไว้ ไม่ใช่จำนวนพร้อมเทรน
ไฟล์ label ว่างคงสถานะเดิม ทั้งที่ตรวจแล้วและที่ยังไม่ตรวจรับ ไม่ได้ใส่ชื่อโรคให้ภาพที่ไม่มีขอบเขต
ชื่อโรคอ้างอิง class ใน label ไม่ใช่การวินิจฉัยหรือการยืนยันโดยผู้เชี่ยวชาญ
รูปต้นฉบับ ไฟล์ YOLO และ class ID ไม่เปลี่ยน และไม่ได้เทรนหรือเปลี่ยน best.pt

ดูภาพพร้อมชื่อ: รัน python -m http.server 8773 --bind 127.0.0.1 จากโฟลเดอร์
temp/rice_labeling แล้วเปิด /rice_named_labels/review.html
ต้องมีโฟลเดอร์ข้อมูลเดิมข้างกันตามพาธใน manifest.json เพื่อเปิดรูปและ label
ภาพ PNG ที่บันทึกจากหน้าดูภาพใช้ตรวจงานเท่านั้น อย่านำภาพที่มีข้อความไปเทรน
''')
    update_viewers()
    target = BASE / 'rice_label_class_names.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                archive.write(p, str(p.relative_to(out)))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert sum(n.startswith('annotations/') for n in archive.namelist()) == len(all_rows)
    print(json.dumps(dict(stage='class_names_complete', **report, zip_bytes=target.stat().st_size)), flush=True)


if __name__ == '__main__':
    publish()
