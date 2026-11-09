"""Publish Tungro segmentation drafts and individually reviewed annotations.

Append Tungro=9, preserve old IDs 0..8. No model is trained or overwritten.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,json,shutil,zipfile
import cv2,numpy as np
from auto_label_rice import ORIGINAL_NAMES,write_json,yolo_text,digest_file
from prepare_rice_training import copy_file
from label_tungro import ROOT,WORK

NAMES=ORIGINAL_NAMES+['Tungro']


def read(p):return json.loads(p.read_text())


def draft_masks(color,sam):
    """Boundary agreement is a draft-selection rule, never disease evidence."""
    if not color:return sam
    if not sam:return color
    def mask(objects):
        a=np.zeros((384,384),np.uint8)
        for o in objects:cv2.fillPoly(a,[np.rint(np.array(o['polygon'])*384).astype(np.int32)],1)
        return a
    support=mask(sam);kept=[]
    for obj in color:
        m=mask([obj])
        if np.count_nonzero(m&support)/max(1,m.sum())>=.6:kept.append(obj)
    if not kept:return sam
    existing=mask(kept)
    for obj in sam:
        m=mask([obj])
        if np.count_nonzero(m&existing)/max(1,m.sum())<.25:
            kept.append(obj);existing|=m
    return kept


def choose(r,decisions):
    color=read(WORK/'proposals'/(r['id']+'.json'))
    sam_path=WORK/'sam_t'/(r['id']+'.json')
    sam=read(sam_path) if sam_path.exists() else None
    decision=decisions.get(r['id'])
    # Uninspected results remain explicitly provisional; no inferred negative.
    objects=draft_masks(color['objects'],(sam or {}).get('objects',[]))
    status='automatic_symptom_mask_draft' if objects else 'needs_actual_annotation'
    if decision:
        if decision['accepted']:
            objects=read(WORK/decision['annotation_source'])['objects']
            if 'selected_objects' in decision:objects=[objects[i] for i in decision['selected_objects']]
            status='ai_visually_reviewed_symptom_mask'
        else:
            objects=[];status='visual_review_requires_reannotation'
    return objects,status,dict(color_proposal=color['objects'],sam_proposal=(sam or {}).get('objects',[]),
        sam_object_support_executed=sam is not None,
        visual_decision=decision,semantic_class_source='unconfirmed_supplied_Tungro_category',
        expert_diagnosis_confirmed=False,model_did_not_diagnose_Tungro=True,
        annotation_target='visible_discolored_leaf_tissue_not_whole_frame')


def build(out):
    if out.exists():raise ValueError('Choose a fresh output to preserve previous editions')
    out.mkdir(parents=True)
    rows=read(WORK/'inventory.json')
    assert read(ROOT/'report.json')['class_names']==ORIGINAL_NAMES
    decisions={x['id']:x for x in read(WORK/'visual_decisions.json')}
    base_train=[r for r in read(ROOT/'manifest.json') if r['default_split']=='train']
    base_val=read(ROOT/'validation_manifest.json')
    val_groups={r['split_group'] for r in base_val};val_sha={r['sha256'] for r in base_val}
    default_train=[];draft_train=[];tungro_reviewed=[];tungro_draft=[];new=[]
    for r in base_train+base_val:
        for field in ['image','label']:copy_file(ROOT/r[field],out/r[field])
    default_train=['./'+r['image'] for r in base_train];draft_train=list(default_train)
    stats=Counter()
    for r in rows:
        objects,status,pred=choose(r,decisions)
        item=dict(r,quality=status,object_count=len(objects),class_ids=[9] if objects else [],
            label='labels/pool/'+r['id']+'.txt' if objects else None,
            prediction_file='predictions/'+r['id']+'.json',export_class_id=9,
            source_class_id=9,
            source_taxonomy_class_id=r['source_class_id'],expert_diagnosis_confirmed=False,
            source_category_expert_verified=False,default_split=None,draft_split=None)
        # Source-category conflicts stay out of both automatic and reviewed pools.
        holds=[x for x in r['reasons'] if x!='nine_class_models_cannot_label_this_new_disease']
        if r['split_group'] in val_groups or r['sha256'] in val_sha:holds.append('related_image_in_validation')
        item['reasons']=holds
        for field in ['image','thumbnail']:copy_file(ROOT/r[field],out/r[field])
        if objects:
            path=out/item['label'];path.parent.mkdir(exist_ok=True,parents=True)
            path.write_text(yolo_text(objects))
            if not holds:
                draft_train.append('./'+r['image']);tungro_draft.append('./'+r['image']);item['draft_split']='train'
                if status=='ai_visually_reviewed_symptom_mask':
                    default_train.append('./'+r['image']);tungro_reviewed.append('./'+r['image']);item['default_split']='train'
        write_json(out/item['prediction_file'],dict(pred,objects=objects,id=r['id']))
        stats[status]+=1;new.append(item)
    val=['./'+r['image'] for r in base_val]
    for name,paths in [('train',default_train),('train_draft',draft_train),('val',val),
        ('tungro_reviewed',tungro_reviewed),('tungro_draft',tungro_draft)]:
        (out/(name+'.txt')).write_text('\n'.join(paths)+'\n')
    for name,train in [('data.yaml','train.txt'),('data_draft.yaml','train_draft.txt')]:
        (out/name).write_text('train: '+train+'\nval: val.txt\nnc: 10\nnames:\n'+
            ''.join('  %d: %s\n'%(i,n) for i,n in enumerate(NAMES)))
    write_json(out/'manifest.json',new);write_json(out/'base_manifest.json',base_train+base_val)
    write_json(out/'validation_manifest.json',base_val)
    exported_decisions=[]
    for d in decisions.values():
        item=dict(d)
        if d['accepted']:
            original=read(WORK/d['annotation_source']);source='accepted_sources/'+d['id']+'.json'
            write_json(out/source,original);item['annotation_source']=source
            item['original_working_source']=d['annotation_source']
        exported_decisions.append(item)
    write_json(out/'visual_decisions.json',exported_decisions)
    write_json(out/'class_mapping.json',dict(names=NAMES,tungro_export_class_id=9,
        source_catalog_tungro_id=19,old_model_class_ids_preserved=list(range(9)),
        cannot_mix_with_22_class_catalog_labels_without_remapping=True))
    report=dict(total_tungro_images=len(rows),status_counts=dict(stats),class_names=NAMES,
        tungro_label_files=sum(bool(r['label']) for r in new),
        tungro_without_accepted_or_draft_mask=sum(not r['label'] for r in new),
        visually_inspected_images=len(decisions),tungro_reviewed_train_images=len(tungro_reviewed),
        tungro_draft_train_images=len(tungro_draft),base_train_images=len(base_train),
        base_train_is_original_checkpoint_training_set=False,
        default_train_images=len(default_train),draft_train_images=len(draft_train),
        validation_images=len(base_val),tungro_validation_images=0,
        validation_does_not_measure_new_Tungro_class=True,all_diagnoses_expert_confirmed=False,
        source_folder_identity_confirmed=False,training_executed=False,
        primary_model_sha256=digest_file(Path('best.pt')),
        object_segmentation_model_sha256=digest_file(Path('temp/rice_labeling/models/sam2.1_t.pt')))
    write_json(out/'report.json',report)
    with (out/'image_status.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','source_relative_path','quality','object_count','label','default_split','draft_split','reasons']
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in new:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],list) else r[k] for k in fields})
    (out/'README_TH.md').write_text('''# Label Tungro (โรคใบสีส้ม)

Tungro ใช้เลขคลาส 9 ต่อจากคลาสเดิม 0–8 ทั้งชุดนี้มี 10 คลาส
ชื่อโรคมาจากหมวดที่ได้รับมา ยังไม่มีผู้เชี่ยวชาญยืนยัน จึงไม่ใช่ expert ground truth
ขอบเขตหมายถึงส่วนเนื้อเยื่อใบที่เห็นสีผิดปกติ ไม่ใช่ขอบทั้งภาพ
SAM เป็นตัวช่วยแยกวัตถุ ไม่ใช่โมเดลที่วินิจฉัย Tungro

manifest.json และ image_status.csv บอกสถานะรูป Tungro ครบทุกภาพ
มีการเปิดภาพจริงตรวจบางรายการ ดู visual_decisions.json
รายการอัตโนมัติที่ยังไม่ตรวจรับคงสถานะ draft ข้อเสนอที่ตรวจแล้วผิดถูกพักไว้
ภาพที่ไม่มีขอบเขตไม่ถูกสร้าง empty label และไม่ถือว่าเป็นภาพข้าวปกติ

data.yaml: ชุดเพิ่มเติมที่จัดไว้ก่อนหน้า 4,209 รูป + Tungro ที่ AI ตรวจรับขอบเขตรายภาพ
data_draft.yaml: ชุดเพิ่มเติมก่อนหน้า + Tungro อัตโนมัติที่ยังต้องตรวจรับ
4,209 รูปนี้ไม่ใช่ชุดฝึกดั้งเดิมของ best.pt และมี label อัตโนมัติอยู่ด้วย
ควรรวมชุดฝึกดั้งเดิมของเพื่อนก่อน fine-tune เพื่อคงข้อมูลของคลาสเดิมให้ครบ
สอง YAML นี้ใช้ val เดิม 34 ภาพที่ AI ตรวจเพียงบางคลาส ซึ่งไม่มี Tungro
จึงยังวัดคุณภาพ Tungro ไม่ได้ และไม่ใช่ชุดประเมินอิสระครบทุกคลาส
ต้องใช้ val/test อิสระที่มี Tungro และแยกภาพต้นฉบับกับภาพดัดแปลงออกจากกัน
ถ้าเพื่อนมีข้อมูลฝึกเดิมครบ ให้ใช้รายการ tungro_reviewed.txt / tungro_draft.txt
รวมกับข้อมูลฝึกเดิมของเพื่อน พร้อมตรวจการซ้ำกับ val/test ของเพื่อนอีกครั้ง
อย่าชี้ train ไปทั้ง images/pool เพราะมีทั้งภาพพักไว้และ val อยู่ด้วย

ไม่ได้เทรนหรือเปลี่ยน best.pt เดิม โมเดลเดิมจึงยังไม่รู้จัก Tungro
weights/best.pt เป็นสำเนา checkpoint เดิมสำหรับส่งต่อ ไม่ใช่โมเดล 10 คลาสที่เทรนแล้ว

เปิดดู: python -m http.server 8772 --bind 127.0.0.1 แล้วเปิด review.html
ตรวจไฟล์: python validate_tungro.py --root . --load-yolo
''',encoding='utf-8')
    shutil.copy2(Path(__file__).with_name('rice_tungro_review.html'),out/'review.html')
    shutil.copy2(Path(__file__).with_name('validate_tungro.py'),out/'validate_tungro.py')
    shutil.copy2(ROOT/'requirements.txt',out/'requirements.txt')
    (out/'weights').mkdir();shutil.copy2(Path('best.pt'),out/'weights/best.pt')
    print(json.dumps(report,ensure_ascii=False),flush=True)


def package(out):
    target=out.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',allowZip64=True) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix!='.cache':z.write(p,str(p.relative_to(out)),
                compress_type=zipfile.ZIP_STORED if p.suffix in ['.jpg','.pt'] else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(json.dumps(dict(stage='tungro_zip_crc_passed',file=str(target),bytes=target.stat().st_size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('temp/rice_labeling/rice_tungro_labeled'))
    a=p.parse_args();build(a.output)
    from validate_tungro import validate
    validate(a.output,load=True);package(a.output)
