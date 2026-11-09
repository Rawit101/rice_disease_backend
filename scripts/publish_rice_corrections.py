"""Publish actual boundary repairs and explicitly provisional visual corrections.

No wholesale class renaming from unverified folders, training, or model updates.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,json,shutil
from auto_label_rice import ORIGINAL_NAMES,write_json,yolo_text,digest_file
from prepare_rice_training import copy_file
from label_rice_archive import package
from validate_rice_archive import validate,validate_loader


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def verify_repairs(base,out):
    old={r['id']:r for r in read(base/'manifest.json')}
    current=read(out/'manifest.json');stats=Counter()
    assert set(old)=={r['id'] for r in current}
    for r in current:
        previous=old[r['id']]
        assert r['label']==previous['label'], 'No invented labels for unannotated images'
        assert r['sha256']==previous['sha256']
        assert (out/r['image']).stat().st_size==(base/r['image']).stat().st_size
        before=read(base/r['prediction_file'])['objects']
        after=read(out/r['prediction_file'])['objects']
        assert len(after)==r['object_count']
        assert {o['class_id'] for o in after}==set(r['class_ids'])
        if not r['repair_types'] or 'ai_visual_annotation' not in r['repair_types']:
            assert {o['class_id'] for o in before}=={o['class_id'] for o in after}
        if r['label']:
            assert (out/r['label']).read_text()==yolo_text(after)
            changed=(out/r['label']).read_bytes()!=(base/r['label']).read_bytes()
            assert changed==r['repair_changed']
            stats['changed_labels']+=changed
        else:
            assert not after and r['default_split'] is None
        if previous['exact_duplicate']:assert r['default_split'] is None
        if 'related_image_in_checked_validation' in previous['reasons']:
            assert r['default_split'] is None
        stats['source_files_checked']+=1
    assert digest_file(base/'weights/best.pt')==digest_file(out/'weights/best.pt')
    result=dict(status='passed',**stats,no_unreviewed_class_changes=True,
        no_new_labels_invented=True,original_checkpoint_preserved=True,
        disease_ground_truth_verified=False)
    write_json(out/'repair_verification.json',result)
    print(json.dumps(result),flush=True)


def build(base,corrections,out):
    if out.exists():raise ValueError('Preserve existing editions; choose a new output directory')
    for d in ['images/pool','labels/pool','predictions','thumbnails','weights']:(out/d).mkdir(parents=True,exist_ok=True)
    manifest=read(base/'manifest.json');val=read(base/'validation_manifest.json')
    old_report=read(base/'report.json')
    decisions=read(corrections/'visual_decisions.json')
    decisions_by_sha={x['sha256']:x for x in decisions}
    accepted={}
    for d in decisions:
        if not d['annotation_source']:continue
        annotation=read(corrections/d['annotation_source']);objects=annotation['objects']
        if d['audit_index']==57:
            objects=[o for o in objects if sum(p[1] for p in o['polygon'])/len(o['polygon'])<.8]
        assert objects
        accepted[d['sha256']]=dict(objects=objects,review=d,
            method=annotation.get('method','visually_selected_refinement_after_source_overlay_review'))
    stats=Counter();new_manifest=[];train=[];strict=[];raw=[]
    for i,old in enumerate(manifest,1):
        r=dict(old);sha=r['sha256'];before=read(base/r['prediction_file'])
        objects=before['objects'];notes=[];revision=[]
        visual=accepted.get(sha) if r['source_class_id']<9 else None
        bpath=corrections/'boundary'/(sha+'.json')
        boundary=read(bpath) if bpath.exists() and r['label'] else None
        if boundary and old['quality']!='ai_visually_screened_mask':
            assert boundary.get('schema')==3
            # Neither color nor contour connectivity establishes the intended
            # object instances. Ultralytics intentionally connects disjoint
            # components with thin bridges. Hold both kinds of proposal until
            # an individual visual annotation is accepted.
            if boundary['geometry_changes']:
                stats['source_files_with_topology_proposal_held']+=1
                notes.append('Disjoint contours may encode one object; automatic instance splitting was held.')
            if boundary['boundary_changes']:
                stats['source_files_with_color_trim_proposal_held']+=1
                notes.append('Color-based trim is an unaccepted proposal; the active label was not changed by color.')
            if any(o['needs_visual_boundary_review'] for o in boundary['object_audit']):
                notes.append('Large image-derived proposals were held; retained the original object where an automatic change was unsafe.')
        if visual:
            objects=visual['objects'];revision.append('ai_visual_annotation')
            r['quality']='ai_visually_screened_mask'
            # These are adjudicated individual image masks, not a blanket folder
            # relabel. Keep source conflict/evaluation holds until independently resolved.
            resolved={'predicted_class_differs_from_source_folder','models_disagree_on_overlapping_object_class',
                      'low_confidence_high_resolution_recovery','visual_review_previously_found_mask_problems'}
            if all(o['class_id']==r['source_class_id'] for o in objects):
                r['ai_resolved_reasons']=[x for x in r['reasons'] if x in resolved]
                r['reasons']=[x for x in r['reasons'] if x not in resolved]
            stats['source_files_with_ai_visual_annotation']+=1
        label=r['label']
        if label:
            text=yolo_text(objects);old_text=(base/label).read_text(encoding='utf-8')
            changed=text!=old_text
            (out/label).write_text(text,encoding='utf-8')
            if changed:stats['label_files_changed']+=1
        else:changed=False
        r.update(object_count=len(objects),class_ids=sorted({o['class_id'] for o in objects}),
            repair_changed=changed,repair_types=revision,repair_notes=notes,
            has_unreviewed_boundary_proposal=bool(boundary and (boundary['geometry_changes'] or boundary['boundary_changes'] or any(o['needs_visual_boundary_review'] for o in boundary['object_audit']))),
            source_category_expert_verified=False,
            visual_repair_review_status=decisions_by_sha.get(sha,{}).get('status','not_visually_reviewed'),
            expert_diagnosis_confirmed=False)
        r['ai_proposed_class_correction']=set(o['class_id'] for o in objects)!=set(o['class_id'] for o in before['objects'])
        if r['ai_proposed_class_correction']:
            stats['source_files_with_individually_proposed_class_correction']+=1
        eligible=r['default_split']=='train'
        if visual and not r['exact_duplicate'] and label and not r['reasons']:
            eligible=all(o['class_id']==r['source_class_id'] for o in objects)
        if any(x=='related_image_in_checked_validation' for x in r['reasons']):eligible=False
        r['default_split']='train' if eligible else None
        if eligible:train.append('./'+r['image'])
        # The strict pool retains only unchanged two-teacher labels. A repaired
        # mask does not acquire two-model agreement merely by being edited.
        r['two_model_agreement_candidate']=bool(old.get('two_model_agreement_candidate') and not changed and eligible)
        if r['two_model_agreement_candidate']:strict.append('./'+r['image'])
        if label and not r['exact_duplicate'] and 'related_image_in_checked_validation' not in r['reasons']:
            raw.append('./'+r['image'])
        for field in ['image','thumbnail']:
            copy_file(base/r[field],out/r[field])
        if changed or visual:
            after=dict(before,objects=objects,objects_before=before['objects'],
                repair_types=revision,repair_notes=notes,
                visual_annotation=visual,
                all_masks_unreviewed=not bool(visual))
            write_json(out/r['prediction_file'],after)
        else:copy_file(base/r['prediction_file'],out/r['prediction_file'])
        new_manifest.append(r)
        if i%2000==0:print(json.dumps(dict(stage='publishing_repairs',completed=i,total=len(manifest))),flush=True)
    for r in val:
        for field in ['image','label']:copy_file(base/r[field],out/r[field])
    for name,rows in [('train',train),('strict_train',strict),('all_prediction_train',raw)]:
        (out/(name+'.txt')).write_text('\n'.join(rows)+'\n',encoding='utf-8')
    files=['val.txt','validation_manifest.json','data.yaml','data_strict.yaml','data_all_predictions.yaml',
        'class_mapping.json','inference_config.json','requirements.txt','restore_rice_images.py',
        'merge_rice_training.py','validate_rice_archive.py','README_LABELS_ONLY.md']
    for name in files:shutil.copy2(base/name,out/name)
    shutil.copy2(base/'weights/best.pt',out/'weights/best.pt')
    report=dict(old_report,default_train_images=len(train),two_model_agreement_train_images=len(strict),
        raw_prediction_train_images=len(raw),status_counts=dict(Counter(r['quality'] for r in new_manifest)),
        source_category_ground_truth='owner_reports_no_expert_confirmation',
        correction_summary=dict(stats),source_images_visually_inspected=len(decisions),
        individually_accepted_visual_annotations=len(accepted),
        all_repairs_are_expert_ground_truth=False,training_executed=False)
    write_json(out/'manifest.json',new_manifest);write_json(out/'report.json',report)
    write_json(out/'repair_summary.json',dict(**stats,visually_inspected_unique_images=len(decisions),
        accepted_visual_annotations=len(accepted),base_manifest_sha256=digest_file(base/'manifest.json'),
        original_images_modified=False,original_checkpoint_modified=False,
        unresolved_disease_classes_not_renamed_from_folder_alone=True))
    write_json(out/'visual_decisions.json',decisions)
    # A compact queue for an expert: the supplied folder names are unverified.
    with (out/'expert_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['id','source_folder','source_relative_path','class_ids',
                            'has_label','default_split','reasons','proposed_class_change'])
        w.writeheader()
        for r in new_manifest:
            if r['reasons'] or not r['label'] or r['ai_proposed_class_correction']:
                w.writerow(dict(id=r['id'],source_folder=r['source_folder'],
                    source_relative_path=r['source_relative_path'],class_ids=json.dumps(r['class_ids']),
                    has_label=bool(r['label']),default_split=r['default_split'],
                    reasons=json.dumps(r['reasons']),proposed_class_change=r['ai_proposed_class_correction']))
    fields=['id','source_folder','quality','repair_changed','repair_types','visual_repair_review_status',
            'default_split','object_count','class_ids','reasons','source','label']
    with (out/'image_status.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in new_manifest:w.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),list) else r.get(k) for k in fields})
    (out/'README_TH.md').write_text('''# ชุดแก้ขอบเขตโรคข้าว

ชุดนี้เป็นฉบับแก้จาก rice_archive_auto ไม่ได้เทรนหรือเปลี่ยน best.pt เดิม
ดูจำนวนแก้จริงใน repair_summary.json และดูรายภาพใน image_status.csv

ตรวจการเชื่อมวงเส้นขอบและข้อเสนอแก้ตามสีครบทุกภาพที่มี label แล้ว
เส้นเชื่อมบางแบบเป็นวิธีเก็บ mask หลายส่วนของวัตถุเดียวตามรูปแบบ YOLO
จึงไม่แยกเป็นวัตถุใหม่อัตโนมัติ และไม่ใช้สีอย่างเดียวระบุบริเวณโรค
ข้อเสนอทั้งสองแบบที่ยังไม่ได้ตรวจรับไม่ถูกนำมาแทน label ที่ใช้งาน
ชื่อโฟลเดอร์ก็ไม่ได้ถูกใช้เปลี่ยนชื่อโรคทุกภาพ

มีการตรวจภาพต้นฉบับและข้อเสนอขอบเขต 290 ภาพ และวาดขอบเขตใหม่บางภาพ
เฉพาะรายการที่ตรวจรับแล้วเท่านั้นถูกแทนที่ ดู visual_decisions.json
การตรวจด้วย AI และชื่อที่ AI เสนอ ยังไม่ใช่การยืนยันโรคจากผู้เชี่ยวชาญ
ผู้ใช้ระบุว่าชื่อโฟลเดอร์ชุด 10 GB ยังไม่มีผู้รู้ตรวจยืนยัน
จึงยังมีภาพที่ชื่อโรค/ส่วนที่เป็นโรคต้องได้รับคำตอบอิสระก่อนใช้เป็น ground truth

data.yaml เป็นรายการฝึกเพิ่มเติมที่ผ่านเกณฑ์เดิมและรายการแก้ที่ยอมรับได้
data_strict.yaml เก็บเฉพาะผลสองโมเดลตรงกันที่ไม่ได้เปลี่ยน label
data_all_predictions.yaml มีผลที่ยังขัดแย้ง ใช้เพื่อซ่อมและตรวจ ห้ามถือว่าถูกทั้งหมด
ให้ใช้รายการ .txt ตาม YAML ไม่ชี้ train ไปทั้ง images/pool
ควรรวมกับข้อมูลฝึกเดิม และใช้ val/test อิสระของเพื่อนในการวัดความแม่นยำ
val 34 ภาพในชุดนี้เป็นภาพที่ AI เคยตรวจเพียงบางคลาส ไม่ใช่ตัววัดครบทุกโรค

เปิดดู: python -m http.server 8771 --bind 127.0.0.1 แล้วเปิด corrections.html
ตรวจโครงสร้าง: python validate_rice_archive.py --root . --load-yolo
แบบ labels_only ต้องมีรูปต้นฉบับชุด 10 GB แล้วใช้ restore_rice_images.py
ตาม README_LABELS_ONLY.md ก่อนตรวจไฟล์หรือเทรน
''',encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('rice_corrections_review.html'),out/'corrections.html')
    print(json.dumps(dict(stage='published',**stats,train_images=len(train)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('temp/rice_labeling/rice_archive_auto'))
    p.add_argument('--corrections',type=Path,default=Path('temp/rice_labeling/archive_corrections'))
    p.add_argument('--output',type=Path,default=Path('temp/rice_labeling/rice_archive_corrected'))
    a=p.parse_args();build(a.base,a.corrections,a.output)
    verify_repairs(a.base,a.output)
    validate(a.output);validate_loader(a.output)
    package(argparse.Namespace(output=a.output.resolve()))
