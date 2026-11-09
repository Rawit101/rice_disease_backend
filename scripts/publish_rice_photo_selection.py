"""Publish visually selected photos without carrying forward old disease masks."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,shutil,zipfile
from screen_rice_image_quality import BASE,OUT
from audit_rice_10class import read,write
from publish_rice_class_names import CLASSES


def csv_file(path,fields,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def publish():
    metrics=read(OUT/'quality_metrics.json');selection=read(OUT/'visual_selection.json')
    decisions=read(OUT/'photo_visual_decisions.json')
    assert len(selection)==len(decisions)==290
    byhash=defaultdict(list)
    for r in metrics:
        if r['pixel_sha256']:byhash[r['pixel_sha256']].append(r)
    inspected=[];decided={}
    for r in selection:
        code,note=decisions[str(r['visual_index'])];assert code in {'K','O','H','D'}
        row=dict(r,decision=code,note=note,source_hint_ids=sorted({x['expected_class_id'] for x in byhash[r['pixel_sha256']]}),
                 disease_confirmed=False,label_ready=False,reviewer='Codex source-photo visual screening',
                 visually_inspected_for_photo_selection=True)
        inspected.append(row);decided[r['pixel_sha256']]=row
    kept=[r for r in inspected if r['decision']=='K'];assert len({r['pixel_sha256'] for r in kept})==len(kept)
    allrows=[]
    for r in metrics:
        d=decided.get(r['pixel_sha256']);status='not_visually_reviewed';note='ยังไม่ได้เปิดตรวจรายภาพ'
        if d:
            status={'K':'selected_photo','O':'omitted_quality','H':'held_unclear_target','D':'omitted_near_duplicate'}[d['decision']]
            note=d['note']
            if d['id']!=r['id'] and d['decision']=='K':status='exact_duplicate_of_selected'
        elif not r['image_decode_ok'] or 'nearly_all_black' in r['metrics_review_flags'] or 'nearly_blank_or_little_visible_content' in r['metrics_review_flags']:
            status='automatic_unusable_content_flag';note='ไฟล์เสียหรือมีเนื้อภาพให้เห็นน้อยมากจากการตรวจพิกเซล ยังไม่ใช่การตรวจชื่อโรค'
        allrows.append(dict(id=r['id'],filename=r['filename'],source_hint=CLASSES[r['expected_class_id']]['name_en'],
                            status=status,visual_representative_id=d['id'] if d else '',
                            pixel_duplicate_group=r.get('pixel_duplicate_group',''),
                            automatic_flags=' | '.join(r['metrics_review_flags']),
                            reviewed_note=note,disease_confirmed=False,label_ready=False))
    report=dict(all_files=len(metrics),reviewed_images=len(inspected),selected_images=len(kept),
                visual_decision_counts=dict(Counter(r['decision'] for r in inspected)),
                all_file_status_counts=dict(Counter(r['status'] for r in allrows)),
                all_files_visually_reviewed=False,exact_decoded_contents=len(byhash),
                exact_duplicate_file_copies=len(metrics)-len(byhash),
                labels_exported=0,source_images_changed=False,
                selected_images_by_unverified_hint={c['name_en']:sum(c['class_id'] in r['source_hint_ids'] for r in kept) for c in CLASSES})
    write(OUT/'photo_selection_report.json',report);write(OUT/'photo_inspection_manifest.json',inspected)
    fields=list(allrows[0]);csv_file(OUT/'photo_screening_all_files.csv',fields,allrows)
    selected_rows=[dict(id=r['id'],filename=r['filename'],source_hint=' | '.join(CLASSES[i]['name_en'] for i in r['source_hint_ids']),
                        source_hint_verified=False,note=r['note'],disease_confirmed=False,label_ready=False) for r in kept]
    selected_fields=list(selected_rows[0]);csv_file(OUT/'selected_images.csv',selected_fields,selected_rows)
    readme='''ภาพข้าวที่คัดก่อนทำ label — รอบตรวจ 3 พฤศจิกายน 2026

ตรวจพิกเซล/ไฟล์ครบ 20,273 ไฟล์ในขอบเขต 9 โรคเดิม + Tungro
เปิดตรวจรายภาพ 290 ภาพ คัดเก็บ 91 ภาพสำหรับทำ annotation ใหม่
รอบนี้ยังไม่ได้เปิดตรวจด้วยตาครบทุกภาพ ส่วนที่เหลือคงสถานะรอตรวจ
ภาพซ้ำแบบพิกเซลตรงกันเก็บเพียงตัวแทน ไม่เพิ่มจำนวนภาพอิสระด้วยสำเนาซ้ำ
ภาพใกล้เคียงที่ตัดออกด้วยตาเป็นการคัดความซ้ำของมุม/ฉบับดัดแปลง ไม่ใช่การยืนยัน parent image ทุกคู่

ภาพคัดเก็บผ่านการดูรายละเอียดของภาพเท่านั้น ยังไม่ได้ยืนยันชื่อโรค
หมวดต้นทางใน CSV เป็นข้อมูลที่ได้รับและอาจขัดกัน ไม่ใช่ ground truth
ชุดนี้ไม่รวม mask/label เดิม ต้องระบุ class ที่ถูกต้องและวาดเฉพาะรอยหรือบริเวณอาการใหม่ก่อนเทรน
ห้ามเอา class จากชื่อไฟล์หรือโฟลเดอร์มาแทน label ที่ยังไม่มี

เปิด index.html เพื่อดูภาพที่คัด ถ้าเบราว์เซอร์จำกัดไฟล์ในเครื่องและมี Python
รัน python3 serve.py (Mac) หรือ py serve.py (Windows) แล้วเปิด http://127.0.0.1:8776/index.html
selected_images.csv คือรายชื่อที่คัดเก็บ; photo_screening_all_files.csv เป็นสถานะทุกรูป
ต้นฉบับและผล label รอบเก่าไม่ถูกลบหรือแก้
'''
    for dirname,items in [('rice_photos_selected',kept),('rice_photo_quality_review',inspected)]:
        dest=BASE/dirname
        if dest.exists():raise ValueError('Preserve prior editions; output already exists: '+str(dest))
        (dest/'images').mkdir(parents=True)
        portable=[]
        for r in items:
            src=BASE/r['dataset_root']/r['image'];name='images/'+r['id']+src.suffix
            shutil.copyfile(src,dest/name)
            assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256((dest/name).read_bytes()).digest()
            portable.append(dict(id=r['id'],filename=r['filename'],image=name,decision=r['decision'],note=r['note'],source_hint_ids=r['source_hint_ids'],pixel_sha256=r['pixel_sha256']))
        (dest/'catalog.js').write_text('window.RicePhotoQuality='+json.dumps(dict(images=portable,classes=CLASSES,report=report),ensure_ascii=False,separators=(',',':'))+';\n')
        shutil.copyfile(Path(__file__).with_name('rice_photo_quality_view.html'),dest/'index.html')
        (dest/'README_TH.txt').write_text(readme,encoding='utf-8-sig')
        for name in ['selected_images.csv','photo_screening_all_files.csv','photo_selection_report.json']:
            shutil.copyfile(OUT/name,dest/name)
        (dest/'serve.py').write_text('''from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
if __name__=='__main__':
 print('http://127.0.0.1:8776/index.html',flush=True)
 ThreadingHTTPServer(('127.0.0.1',8776),partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).resolve().parent))).serve_forever()
''')
        assert not list(dest.rglob('labels*')) and not list((dest/'images').glob('*.txt'))
        if dirname=='rice_photos_selected':
            target=dest.with_suffix('.zip')
            with zipfile.ZipFile(target,'w',allowZip64=True) as z:
                for p in sorted(dest.rglob('*')):
                    if p.is_file():z.write(p,dirname+'/'+str(p.relative_to(dest)),compress_type=zipfile.ZIP_STORED if p.suffix=='.jpg' else zipfile.ZIP_DEFLATED)
            with zipfile.ZipFile(target) as z:
                assert z.testzip() is None
                assert sum(x.startswith(dirname+'/images/') for x in z.namelist())==len(kept)
            report['selected_zip_bytes']=target.stat().st_size
    write(OUT/'photo_selection_report.json',report)
    print(json.dumps(dict(stage='photo_selection_published',**report)),flush=True)


if __name__=='__main__':publish()
