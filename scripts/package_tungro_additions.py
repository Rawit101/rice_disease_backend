"""Portable Tungro-only annotation package for merging into an existing dataset."""
from pathlib import Path
import argparse,json,zipfile


def package(root):
    read=lambda n:json.loads((root/n).read_text())
    report=read('report.json');manifest=read('manifest.json')
    assert read('validation_result.json')['structural_validation']=='passed'
    assert report['class_names'][9]=='Tungro'
    target=root.parent/'rice_tungro_additions.zip'
    keys=['total_tungro_images','status_counts','class_names','tungro_label_files',
          'tungro_without_accepted_or_draft_mask','visually_inspected_images',
          'tungro_reviewed_train_images','tungro_draft_train_images',
          'all_diagnoses_expert_confirmed','source_folder_identity_confirmed','training_executed']
    compact={k:report[k] for k in keys};compact['package_scope']='Tungro additions only; merge with original training dataset'
    selected=set(['manifest.json','class_mapping.json','visual_decisions.json','image_status.csv',
                  'review.html','tungro_reviewed.txt','tungro_draft.txt',
                  'class_names_th.json','classes.csv','image_classes.csv'])
    for r in manifest:
        for key in ['image','thumbnail','prediction_file','label']:
            if r[key]:selected.add(r[key])
    selected.update(str(p.relative_to(root)) for p in (root/'accepted_sources').glob('*.json'))
    text=f'''# ชุด label เพิ่ม Tungro สำหรับส่งให้เพื่อน

มีรูป Tungro {len(manifest):,} ภาพ พร้อม label {report['tungro_label_files']:,} ไฟล์
ภาพที่ยังไม่มี label ใช้งาน {report['tungro_without_accepted_or_draft_mask']} ภาพ ไม่ใช่ภาพปกติ/negative
AI ตรวจรับขอบเขตรายภาพ {report['status_counts']['ai_visually_reviewed_symptom_mask']} ภาพ
ที่เหลือเป็น label อัตโนมัติแบบร่าง ชื่อโรคยังไม่มีผู้รู้ยืนยัน

Tungro = เลขคลาส 9 ต่อจากคลาสเดิม 0–8 รวมเป็น 10 คลาส
ดู classes.txt, classes.yaml และ class_mapping.json ก่อนรวมข้อมูล
classes.csv / class_names_th.json มีชื่อไทย–อังกฤษ ส่วน image_classes.csv ระบุชื่อ class ของแต่ละรูป
review.html แสดงชื่อโรคและเลข class บนขอบเขตทุกภาพที่มี label
อย่าเปลี่ยนเลขเป็น 0 หรือเอาเลขคลาสจาก catalog 22 ชื่อมาปะปน
รูปและ label ใช้ชื่อรหัสเดียวกันใน images/pool และ labels/pool
manifest.json / image_status.csv จับคู่กลับไปชื่อรูปต้นฉบับได้

tungro_reviewed.txt คือรายการเพิ่มที่ AI ตรวจขอบเขตแล้วและไม่ติดประเด็นชื่อหมวด
tungro_draft.txt รวมผลอัตโนมัติที่ยังต้องตรวจรับ ไม่ใช่ expert ground truth
ให้เพื่อนรวมรายการเพิ่มนี้กับชุดฝึกดั้งเดิมของเพื่อน และตั้ง nc: 10
ไม่ควรเทรนต่อด้วย Tungro อย่างเดียวหากต้องการคงความสามารถของคลาสเดิม
ต้องใช้ val/test อิสระที่มี Tungro และกันภาพต้นฉบับ/ภาพดัดแปลงไม่ให้ข้ามชุด
ไฟล์นี้ไม่มี val/test และไม่มีโมเดลที่เทรนเพิ่มแล้ว

เปิดดู: python -m http.server 8772 --bind 127.0.0.1 แล้วเปิด review.html
เลือก “ยังไม่มี label ที่ใช้งาน” เพื่อดูภาพที่ยังแก้ขอบเขตไม่ได้
'''
    with zipfile.ZipFile(target,'w',allowZip64=True) as z:
        for name in sorted(selected):
            p=root/name;assert p.is_file()
            z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix=='.jpg' else zipfile.ZIP_DEFLATED)
        z.writestr('report.json',json.dumps(compact,ensure_ascii=False,indent=2),compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('README_TH.md',text,compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('classes.txt','\n'.join(report['class_names'])+'\n')
        z.writestr('classes.yaml','nc: 10\nnames:\n'+''.join('  %d: %s\n'%(i,n) for i,n in enumerate(report['class_names'])))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))
        assert sum(n.startswith('images/pool/') for n in names)==len(manifest)
        assert sum(n.startswith('labels/pool/') for n in names)==report['tungro_label_files']
    print(json.dumps(dict(stage='tungro_additions_crc_passed',file=str(target),bytes=target.stat().st_size)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('temp/rice_labeling/rice_tungro_labeled_v2'))
    a=p.parse_args();package(a.root)
