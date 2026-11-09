"""Portable, scoped review queue with explicit audit status and no train split."""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json, shutil, zipfile
from publish_rice_10class_review import BASE, OUT

SCRIPTS=Path(__file__).resolve().parent


def viewer():
    html=(SCRIPTS/'rice_friend_review.html').read_text()
    html=html.replace('ตรวจ label โรคข้าว — ชุดสำหรับเพื่อน','ตรวจทบทวน label — เฉพาะ 10 คลาส')
    html=html.replace('<h1>ตรวจ label โรคข้าว</h1>','<h1>ตรวจทบทวน label — 10 คลาส</h1>')
    html=html.replace('<main><div class="controls">','''<main><div class="note"><strong>เฉพาะ 9 โรคเดิม + Tungro · ชุดตรวจ ยังไม่ใช่ชุดพร้อมเทรน</strong><br>ตรวจความเสี่ยงครบ 20,273 ภาพ เปิดเทียบภาพจริง 122 ภาพ ไม่ใช่การตรวจด้วยตาครบทุกภาพ<br>ชื่อหมวดต้นทางยังไม่ยืนยัน มาตรฐาน: ล้อมเฉพาะรอยหรือบริเวณอาการที่เห็น ไม่ครอบเนื้อใบปกติ</div><div class="controls">''')
    html=html.replace('<select id="filter"', '''<select id="audit" aria-label="สถานะตรวจคุณภาพ"><option value="">ทุกสถานะคุณภาพ</option><option value="reannotation_required">เปิดดูแล้ว ต้องแก้ขอบเขต</option><option value="held_conflict_or_geometry">พักไว้: class/ข้อมูล/ขอบเขตขัดกัน</option><option value="target_definition_pending">รอกำหนดว่าจะล้อมทั้งใบหรือรอย</option><option value="tungro_identity_and_boundary_pending">Tungro รอยืนยันชื่อและขอบเขต</option><option value="corrected_boundary_requires_class_confirmation">เสนอแก้ขอบเขตแล้ว รอยืนยันชื่อ</option><option value="draft_requires_class_and_boundary_confirmation">ฉบับร่าง รอตรวจชื่อและขอบเขต</option><option value="needs_annotation">ยังไม่มี mask ต้องทำเพิ่ม</option></select><select id="filter"''')
    html=html.replace('<p id="status"></p>', '<p id="status"></p><div class="note" id="auditInfo"></div><details><summary>รายละเอียดการเปิดดูภาพรอบนี้</summary><p id="visualNote"></p></details>')
    html=html.replace('<h2>Label พร้อมชื่อโรคและเลข class</h2>', '<h2 id="maskTitle">ขอบเขตเพื่อทบทวน</h2>')
    html=html.replace('<label>ผลตรวจภาพนี้', '<label><input type="checkbox" id="classVerified"> ตรวจชื่อโรคของทุกขอบเขตแล้ว</label><label><input type="checkbox" id="boundaryVerified"> ตรวจความครบและขอบเขตตามเป้าหมายแล้ว</label><label>ผลตรวจภาพนี้')
    html=html.replace("const storageKey=", """const auditNames={needs_annotation:'ยังไม่มี mask ต้องทำ annotation เพิ่ม',reannotation_required:'เปิดดูแล้วพบปัญหา ต้องแก้ขอบเขต',held_conflict_or_geometry:'พักไว้: class/ข้อมูลต้นทาง/ขอบเขตขัดกัน',target_definition_pending:'รอกำหนดเป้าหมายทั้งใบหรือบริเวณอาการ',tungro_identity_and_boundary_pending:'Tungro รอยืนยันชื่อโรคและขอบเขต',draft_requires_class_and_boundary_confirmation:'ฉบับร่าง ยังไม่ตรวจรับชื่อโรคและขอบเขต',corrected_boundary_requires_class_confirmation:'เสนอแก้ขอบเขตตกหล่นแล้ว ยังต้องยืนยันชื่อโรค'};
function auditDetails(r){$('auditInfo').textContent=(auditNames[r.review_status]||r.review_status)+' · หมวดต้นทาง (ยังไม่ยืนยัน): '+RiceClassNames.name(r.expected_class_id,names);$('visualNote').textContent=r.visual_review?('ผลตรวจ label เดิม: '+r.visual_review.note+(r.boundary_correction_proposed?' · เพิ่มขอบเขตที่ตกหล่นแล้ว ดูฉบับเสนอแก้บนภาพ':'') ):'ยังไม่ได้เปิดตรวจภาพนี้รายภาพในรอบนี้';$('maskTitle').textContent=!r.object_count?'ยังไม่มี mask ใช้งาน':r.review_status==='reannotation_required'||r.review_status==='held_conflict_or_geometry'?'Label เดิมที่พักไว้เพื่อแก้ไข':r.boundary_correction_proposed?'ขอบเขตฉบับเสนอแก้ รอยืนยันชื่อโรค':'ขอบเขตฉบับร่าง รอตรวจชื่อและเป้าหมาย'}
const storageKey=""")
    html=html.replace("decision:r.decision,notes:","decision:r.decision,class_verified:r.class_verified===true,boundary_verified:r.boundary_verified===true,notes:")
    html=html.replace("const decision=$('decision').value,notes=$('notes').value;", "const decision=$('decision').value,notes=$('notes').value;if(decision==='accepted'&&(!r.object_count||!$('classVerified').checked||!$('boundaryVerified').checked)){$('decision').value=records[r.id]?.decision||'pending';$('saveInfo').textContent='ยังรับเป็นถูกต้องไม่ได้: ต้องมี mask และตรวจยืนยันชื่อโรคกับขอบเขตทั้งสองส่วนก่อน';return}")
    html=html.replace("records[r.id]={decision,notes,updated_at:", "records[r.id]={decision,notes,class_verified:$('classVerified').checked,boundary_verified:$('boundaryVerified').checked,updated_at:")
    html=html.replace("rows=all.filter(r=>", "rows=all.filter(r=>(!$('audit').value||r.review_status===$('audit').value)&&")
    html=html.replace("${qualityNames[r.quality]||r.quality}", "สถานะจากรอบเก่า: ${qualityNames[r.quality]||r.quality}")
    html=html.replace("$('source').href=r.image;$('label').href=r.label;", "auditDetails(r);$('source').href=r.image;if(r.label){$('label').hidden=false;$('label').href=r.label}else{$('label').hidden=true;$('label').removeAttribute('href')}")
    html=html.replace("$('notes').value=record.notes;", "$('notes').value=record.notes;$('classVerified').checked=record.class_verified===true;$('boundaryVerified').checked=record.boundary_verified===true;")
    html=html.replace("['decision','notes']", "['decision','notes','classVerified','boundaryVerified']")
    html=html.replace("if(!rows.length){$('filename').textContent=", "if(!rows.length){$('auditInfo').textContent='';$('visualNote').textContent='';$('maskTitle').textContent='ไม่มีภาพตรงกับตัวกรอง';$('classVerified').checked=false;$('boundaryVerified').checked=false;$('filename').textContent=")
    html=html.replace("ไม่มีขอบเขตโรคในไฟล์ label นี้", "ยังไม่มี mask ใช้งาน ไม่ถือเป็นภาพปกติ")
    html=html.replace("ไฟล์ label ว่าง ไม่ระบุชื่อโรค", "ยังไม่มี mask ใช้งาน ต้องทำ annotation เพิ่ม")
    html=html.replace("['dataset','class','filter']", "['dataset','class','filter','audit']")
    html=html.replace("$('decision').onchange=save;", "$('classVerified').onchange=save;$('boundaryVerified').onchange=save;$('decision').onchange=save;")
    html=html.replace("'notes','updated_at'];", "'notes','updated_at','source_hint','audit_status','class_verified','boundary_verified'];")
    html=html.replace("record.notes,record.updated_at])", "record.notes,record.updated_at,r.expected_class_name,r.review_status,record.class_verified===true,record.boundary_verified===true])")
    html=html.replace("${all.length.toLocaleString()} ภาพที่มีไฟล์ label", "${all.length.toLocaleString()} ภาพใน 10 หมวดเป้าหมาย")
    html=html.replace("bg.textContent='ไฟล์ label ว่าง / พื้นหลัง'", "bg.textContent='ยังไม่มี mask ต้องทำเพิ่ม'")
    html=html.replace("const auditNames=", """const auditFlagNames={annotation_class_disagrees_with_source_hint:'class ใน label ขัดกับหมวดต้นทาง',no_mask_for_source_hint_class:'ยังไม่มี mask สำหรับ class ตามหมวดต้นทาง',mask_overlaps_near_uniform_background:'mask อาจกินพื้นหลังสีขาว/ดำ',green_tissue_dominates_mask_requires_symptom_boundary_review:'mask มีเนื้อใบสีเขียวมาก ต้องตรวจตามเกณฑ์เฉพาะรอย',exact_image_has_conflicting_source_categories:'ภาพเดียวกันอยู่คนละชื่อหมวด',near_duplicate_source_category_conflict:'ภาพคล้ายกันอยู่คนละชื่อหมวด',previous_visual_mask_problem:'เคยพบปัญหาขอบเขต',multiple_disease_classes_require_confirmation:'มีหลาย class ต้องตรวจชื่อแต่ละบริเวณ',tungro_diagnosis_not_established_by_color_or_sam:'สี/SAM ยังยืนยัน Tungro ไม่ได้',no_usable_mask_not_a_negative:'ยังไม่มี mask ไม่ถือเป็นภาพปกติ',teacher_class_disagreement:'สองโมเดลให้ class ต่างกัน',low_confidence_recovery:'ผลตรวจซ้ำมีความมั่นใจต่ำ'};
const auditNames=""")
    html=html.replace("RiceClassNames.name(r.expected_class_id,names);$('visualNote')", "RiceClassNames.name(r.expected_class_id,names)+' · ที่ต้องตรวจ: '+r.audit_flags.filter(f=>auditFlagNames[f]).slice(0,3).map(f=>auditFlagNames[f]).join(' / ');$('visualNote')")
    return html


def build(refresh=False):
    source=BASE/'rice_10class_review';out=BASE/'rice_friend_review_10class'
    if out.exists() and not refresh:raise ValueError('Use --refresh to update this review edition')
    for name in ['images','draft_labels','held_labels','annotations']:(out/name).mkdir(parents=True,exist_ok=True)
    if refresh:
        # Only replace generated review copies; source annotations are never removed.
        for folder in ['draft_labels','held_labels']:
            for old in (out/folder).glob('*.txt'):old.unlink()
    rows=json.loads((source/'manifest.json').read_text());classes=json.loads((source/'class_names_th.json').read_text())
    assert len(classes)==10 and classes[9]['name_en']=='Tungro'
    assert len(rows)==20273 and all(0<=r['expected_class_id']<10 for r in rows)
    report=json.loads((source/'report.json').read_text())
    fingerprint=hashlib.sha256(report['target_policy'].encode());portable=[];checks=[];labels=0
    for i,r in enumerate(rows,1):
        image=(source/r['image']).resolve();data=json.loads((source/r['annotation']).read_text())
        objects=data['objects'];label=None;sha=None
        image_name='images/'+r['id']+image.suffix
        if refresh and (out/image_name).exists():assert (out/image_name).stat().st_size==image.stat().st_size
        else:shutil.copyfile(image,out/image_name)
        if r['label']:
            original=(source/r['label']).resolve();raw=original.read_bytes();sha=hashlib.sha256(raw).hexdigest()
            assert sha==data['label_sha256'] and objects
            folder='draft_labels' if r['review_status'] in ['draft_requires_class_and_boundary_confirmation','corrected_boundary_requires_class_confirmation'] else 'held_labels'
            label=folder+'/'+r['id']+'.txt';shutil.copyfile(original,out/label);labels+=1;checks.append((original,out/label,sha))
        else:assert not objects
        assert sorted({o['class_id'] for o in objects})==r['class_ids']
        annotation='annotations/'+r['id']+'.js'
        (out/annotation).write_text('window.RiceReviewAnnotations['+json.dumps(r['id'])+']='+json.dumps(dict(objects=objects),ensure_ascii=False,separators=(',',':'))+';\n')
        portable.append(dict(r,image=image_name,label=label,annotation=annotation))
        fingerprint.update((r['id']+':'+str(sha)+'\n').encode())
        if i%2500==0:print(json.dumps(dict(stage='copy_10class_review',completed=i,total=len(rows))),flush=True)
    catalog=dict(manifest=portable,classes=classes,fingerprint=fingerprint.hexdigest())
    (out/'catalog.js').write_text('window.RiceReviewCatalog='+json.dumps(catalog,ensure_ascii=False,separators=(',',':'))+';\n')
    (out/'index.html').write_text(viewer());shutil.copyfile(SCRIPTS/'rice_class_names.js',out/'rice_class_names.js')
    for name in ['classes.csv','image_classes.csv','report.json','visual_decisions.json']:
        shutil.copyfile(source/name,out/name)
    shutil.copyfile(SCRIPTS.parent/'TEN_CLASS_POLICY.md',out/'TEN_CLASS_POLICY.md')
    shutil.copyfile(OUT/'excluded_outside_scope.csv',out/'excluded_outside_scope.csv')
    text='''เว็บตรวจทบทวนเฉพาะ 10 คลาส: 9 โรคเดิม + Tungro

แตก ZIP ทั้งหมด แล้วเปิด rice_friend_review_10class/index.html ด้วยเบราว์เซอร์
เลือกสถานะคุณภาพเพื่อดูภาพที่พักไว้/ต้องแก้/ยังไม่มี mask
หมวดต้นทางเป็นเพียงข้อมูลที่ได้รับ ยังไม่ได้ยืนยันชื่อโรค
ภาพมีทั้งร่างและ label ที่พบข้อผิดพลาด ชุดนี้ยังไม่ใช่ชุดพร้อมเทรน
draft_labels/ เป็นร่างเพื่อให้ตรวจ ไม่ใช่ label ที่รับรองความถูกต้อง
held_labels/ เป็น label เดิมที่พักไว้ ห้ามรวมเข้าเทรนโดยถือว่าตรวจผ่านแล้ว
863 ภาพไม่มี mask และไม่มีไฟล์ label ว่างสร้างขึ้นแทน
มาตรฐาน: ล้อมเฉพาะรอยหรือบริเวณอาการที่เห็น ไม่ครอบเนื้อใบปกติ

ก่อนเลือก เห็นว่าถูกต้อง ให้ตรวจทั้งชื่อโรคและความครบของขอบเขต แล้วติ๊กสองช่อง
ผลตรวจไม่แก้ไฟล์ label อัตโนมัติ ส่งออก CSV และสำรอง JSON ส่งคืนเจ้าของงาน
ผลตรวจเก็บในเบราว์เซอร์เครื่องนี้ ต้องส่งออกก่อนย้ายเครื่องหรือล้างข้อมูลเบราว์เซอร์

วิธีสำรองถ้าเบราว์เซอร์บล็อกไฟล์ในเครื่อง และมี Python:
เปิด Terminal ในโฟลเดอร์นี้ รัน python3 serve.py (Mac) หรือ py serve.py (Windows)
เปิด http://127.0.0.1:8775/index.html และเปิด Terminal ค้างไว้
ส่งให้เพื่อนเป็น ZIP ทั้งชุด ไม่ส่งลิงก์ 127.0.0.1

รายละเอียดการตรวจ:
'''+(source/'README_TH.txt').read_text(encoding='utf-8-sig')
    (out/'README_TH.txt').write_text(text,encoding='utf-8-sig')
    (out/'serve.py').write_text('''from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
if __name__=='__main__':
    print('http://127.0.0.1:8775/index.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8775),partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).resolve().parent))).serve_forever()
''')
    assert labels==19410 and len(rows)-labels==863
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==sha and hashlib.sha256(q.read_bytes()).hexdigest()==sha for p,q,sha in checks)
    assert not list(out.glob('train*.txt')) and not (out/'labels').exists()
    target=out.with_suffix('.zip');files=sorted(p for p in out.rglob('*') if p.is_file())
    with zipfile.ZipFile(target,'w',allowZip64=True) as z:
        for i,p in enumerate(files,1):
            z.write(p,out.name+'/'+str(p.relative_to(out)),compress_type=zipfile.ZIP_STORED if p.suffix=='.jpg' else zipfile.ZIP_DEFLATED)
            if i%15000==0:print(json.dumps(dict(stage='pack_10class_review',completed=i,total=len(files))),flush=True)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        entries=z.namelist();assert len(entries)==len(set(entries))
        assert sum(n.startswith(out.name+'/images/') for n in entries)==len(rows)
        assert sum(n.startswith(out.name+'/draft_labels/') or n.startswith(out.name+'/held_labels/') for n in entries)==labels
        assert sum(n.startswith(out.name+'/annotations/') for n in entries)==len(rows)
    print(json.dumps(dict(stage='ten_class_review_zip_crc_passed',zip=str(target),bytes=target.stat().st_size,images=len(rows),candidate_labels=labels,not_a_training_dataset=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args();build(args.refresh)
