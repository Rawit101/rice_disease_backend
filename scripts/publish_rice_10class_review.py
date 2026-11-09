"""Publish the ten-class audit as a review queue, never as verified training data."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json
from audit_rice_10class import BASE, OUT, read, write
from publish_rice_class_names import NAMES, CLASSES, parse_label

TARGET_POLICY = 'visible_symptom_regions_only'
AMBIGUOUS_WHOLE_LEAF = {8,13,14,25,27,35,43,44,45,47,88,108}
HOLD_FLAGS = {'green_tissue_dominates_mask_requires_symptom_boundary_review', 'annotation_class_disagrees_with_source_hint', 'exact_image_has_conflicting_source_categories',
              'near_duplicate_source_category_conflict', 'previous_visual_mask_problem',
              'teacher_class_disagreement', 'multiple_disease_classes_require_confirmation',
              'mask_overlaps_near_uniform_background', 'image_decode_error',
              'invalid_label_geometry_or_class', 'no_mask_for_source_hint_class',
              'most_image_is_masked_requires_target_review', 'very_large_mask_requires_target_review'}


def publish():
    out=BASE/'rice_10class_review';(out/'annotations').mkdir(parents=True,exist_ok=True)
    audit=read(OUT/'audit_manifest.json');selection=read(OUT/'visual_selection.json')
    notes=read(OUT/'visual_notes_by_index.json');visual={}
    for entry in selection:
        code,note=notes[str(entry['index'])]
        if entry['index'] in AMBIGUOUS_WHOLE_LEAF and TARGET_POLICY.startswith('pending'):
            code='target_definition_requires_confirmation'
            note='Whole-leaf versus localized-symptom target policy must be agreed before judging this contour. '+note
        visual[entry['id']]=dict(**entry,review_code=code,note=note,
                                disease_identity_verified=False,reviewer='Codex visual audit',
                                review_scale='paired source and overlay contact sheet; selected examples enlarged')
    # An exact duplicate inherits a rejection/hold, never an acceptance claim.
    bysha={r['sha256']:visual[r['id']] for r in audit if r['id'] in visual and
           visual[r['id']]['review_code'] in ['reannotation_required','target_definition_requires_confirmation']}
    proposed={p.stem:read(p) for p in (OUT/'corrections_proposed').glob('*.json')}
    proposed_sha={r['sha256']:proposed[r['id']] for r in audit if r['id'] in proposed}
    manifest=[];human=[];hashes=[];corrected=0
    for r in audit:
        root=BASE/r['dataset_root'];label=(root/r['label']) if r.get('label') else None
        objects=parse_label(label.read_text(),NAMES) if label and 'invalid_label_geometry_or_class' not in r['audit_flags'] else []
        decision=visual.get(r['id'])
        inherited=False
        if decision is None and r['sha256'] in bysha:
            decision=bysha[r['sha256']];inherited=True
        if not objects:status='needs_annotation'
        elif decision and decision['review_code']=='reannotation_required':status='reannotation_required'
        elif decision and decision['review_code']=='target_definition_requires_confirmation':status='target_definition_pending'
        elif HOLD_FLAGS.intersection(r['audit_flags']):status='held_conflict_or_geometry'
        elif r['expected_class_id']==9:status='tungro_identity_and_boundary_pending'
        else:status='draft_requires_class_and_boundary_confirmation'
        original_label=label
        correction=proposed_sha.get(r['sha256']) if r['class_ids']==[4] else None
        if correction:
            assert {o['class_id'] for o in correction['objects']}=={4}
            objects=correction['objects'];label=OUT/'corrections_proposed'/(correction['id']+'.txt')
            status='corrected_boundary_requires_class_confirmation';corrected+=1
        current_sha=hashlib.sha256(label.read_bytes()).hexdigest() if label and objects else None
        source_label='../'+str(label.relative_to(BASE)) if label and objects else None
        annotation='annotations/'+r['id']+'.json'
        write(out/annotation,dict(id=r['id'],objects=objects,label_sha256=current_sha,
                                  label_source=str(label.relative_to(BASE)) if label else None,
                                  expected_class_id=r['expected_class_id'],review_status=status,
                                  source_category_is_unverified=True,expert_diagnosis_confirmed=False))
        row=dict(id=r['id'],dataset=r['dataset'],filename=Path(r['source']).name,
                 image='../'+str((root/r['image']).relative_to(BASE)),label=source_label,annotation=annotation,
                 class_ids=r['class_ids'],object_count=len(objects),quality=r['quality'],split=None,
                 exact_duplicate=r.get('exact_duplicate',False),expected_class_id=r['expected_class_id'],
                 expected_class_name=r['expected_class_name'],review_status=status,audit_flags=r['audit_flags'],
                 visual_review=decision,visual_hold_inherited_from_exact_duplicate=inherited,
                 boundary_correction_proposed=bool(correction),
                 training_ready=False,source_category_is_unverified=True)
        manifest.append(row)
        human.append(dict(id=r['id'],filename=row['filename'],dataset=r['dataset'],
                          source_hint=NAMES[r['expected_class_id']],source_hint_verified=False,
                          candidate_class_ids=' | '.join(map(str,r['class_ids'])),
                          candidate_class_names=' | '.join(NAMES[i] for i in r['class_ids']),
                          review_status=status,flags=' | '.join(r['audit_flags']),
                          visual_review_code=decision['review_code'] if decision else '',
                          visual_note=decision['note'] if decision else '',training_ready=False))
        if original_label and r['label_sha256']:hashes.append((original_label,r['label_sha256']))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes)
    assert len(manifest)==20273 and all(0<=r['expected_class_id']<10 for r in manifest)
    write(out/'manifest.json',manifest);write(out/'class_names_th.json',CLASSES)
    write(out/'visual_decisions.json',list(visual.values()))
    report=dict(read(OUT/'audit_report.json'),review_status_counts=dict(Counter(r['review_status'] for r in manifest)),
                visually_inspected_source_images=len(visual),visual_review_is_a_risk_stratified_sample=True,
                visual_review_code_counts=dict(Counter(d['review_code'] for d in visual.values())),
                source_unique_contents_by_requested_class={NAMES[i]:len({r['sha256'] for r in audit if r['expected_class_id']==i}) for i in range(10)},
                target_policy=TARGET_POLICY,source_label_bytes_unchanged=True,
                proposed_boundary_correction_source_files=corrected,
                verified_training_images=0,is_training_dataset=False)
    write(out/'report.json',report)
    for filename,fields,data in [('classes.csv',['class_id','name_en','name_th'],CLASSES),
                                 ('image_classes.csv',list(human[0]),human)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(data)
    (out/'README_TH.txt').write_text('''ชุดตรวจทบทวนเฉพาะ 10 คลาส — ยังไม่ใช่ชุดพร้อมเทรน

คลาส 0–8 คือ 9 โรคเดิม และ class 9 = Tungro (โรคใบสีส้ม)
กรองด้วยชื่อหมวดต้นทางตามขอบเขตที่เจ้าของงานขอ ไม่ใช้เลข catalog 9 ของ Bakanae แทน Tungro
Leaf Smut ไม่ใช่ False Smut; Sheath Blight ไม่ใช่ Sheath Rot; หมวดอื่นไม่ถูกนำมารวมเป็น label
ชื่อหมวดต้นทางยังไม่ผ่านการยืนยัน จึงใช้เพียงตั้งขอบเขตและแจ้งความขัดแย้ง

ตรวจโครงสร้างและความเสี่ยงของภาพในขอบเขตครบ 20,273 ไฟล์
มี mask เดิม 19,410 ไฟล์ อีก 863 ภาพยังไม่มี mask ใช้งาน ไม่ใช่ negative
เปิดเทียบ source/overlay 122 ภาพที่เลือกตาม class และความเสี่ยง ไม่ใช่การตรวจด้วยตาครบทุกภาพ
ตัวเลขความเสี่ยงไม่ใช่จำนวนที่ยืนยันว่าเป็นโรคผิด และกลุ่มตัวอย่างนี้ใช้ประมาณอัตราผิดทั้งชุดไม่ได้
มีทั้ง mask กินพื้นหลัง เลือกนิ้วมือ รอยตกหล่น และมาตรฐานล้อมทั้งใบ/รอยโรคที่ยังปนกัน
ตรวจจาก outline-only ซ้ำในกลุ่ม Tungro เพื่อไม่ตัดสินผิดเพราะสี fill ของ overlay เปลี่ยนสีใบ

ภาพตัวอย่าง train_batch18270.jpg ในโมเดลเดิมมี mask ที่ล้ำพื้นหลังให้เห็นด้วย
โมเดลเดิมและการเห็นตรงกันของสองเวอร์ชันจึงไม่ใช่ ground truth อิสระ
Tungro ใช้สีเหลือง/ส้มและ SAM ยืนยันโรคไม่ได้ ต้องยืนยันชื่อและขอบเขตทั้งสองส่วน
เจ้าของงานยืนยันใช้ขอบเขตเฉพาะรอยหรือบริเวณอาการที่เห็น ไม่ครอบเนื้อใบปกติ
ข้อมูลเดิมและ best.pt ไม่ถูกแก้ และไม่ได้เทรนโมเดล

เกณฑ์อ้างอิง: IRRI Diagnosis of Common Diseases of Rice
https://www.knowledgebank.irri.org/images/docs/diagnostic-of-common-diseases-of-rice.pdf
''',encoding='utf-8-sig')
    print(json.dumps(dict(stage='ten_class_review_published',**report)),flush=True)


if __name__=='__main__':publish()
