# -*- coding: utf-8 -*-
"""Publish a local, read-only gallery of the fixed 308-image unresolved cohort."""
from pathlib import Path
from collections import Counter
import csv,json,shutil,argparse
from auto_label_rice import write_json
from prepare_rice_training import copy_file


def publish(source,out):
    rows=json.loads((out/'selection.json').read_text())
    results=json.loads((out/'results.json').read_text())
    (out/'images').mkdir(exist_ok=True);(out/'candidates').mkdir(exist_ok=True)
    counts=Counter(r['sha256'] for r in rows)
    approved_path=out/'manual_accepted.json'
    approved={r['source_sha256']:r for r in json.loads(approved_path.read_text())
              if r.get('visually_accepted')} if approved_path.exists() else {}
    audit_path=out/'candidate_audit/review_decisions.json'
    audits=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    manifest=[]
    for r in rows:
        result=results[r['sha256']];q=result['quality'];status=result['status']
        raw_objects=result['objects']
        if r['sha256'] in approved:
            result={**result,'status':'resolved_after_manual_fix',
                    'objects':approved[r['sha256']]['objects'],
                    'raw_crop_objects':raw_objects,'is_training_label':True,
                    'visual_review':approved[r['sha256']]}
            status=result['status']
        reasons={
            'unsupported_new_class':'โมเดลเดิมมี 9 คลาส และยังไม่มีชื่อโรคนี้ จึงต้องทำ annotation ของโรคใหม่ก่อน ไม่ใช่การยืนยันว่ารูปใช้ไม่ได้',
            'nearly_empty_black_crop':'ต้นฉบับเกือบเป็นสีดำทั้งหมด แทบไม่เห็นวัตถุที่จะล้อมขอบเขต ควรหาไฟล์ก่อนตัดพื้นหลังมาแทน',
            'crop_candidates_need_review':'ซูมแล้วพบกรอบเสนอใหม่ แต่ยังต้องตรวจชื่อโรค ความครบ และขอบเขตที่อาจถูกขอบภาพตัด ก่อนบันทึกเป็น label',
            'no_usable_crop_prediction':'ตรวจทั้งภาพและซูมเพิ่มเติมแล้ว ยังไม่พบขอบเขตที่โมเดลเสนอได้ ไม่ได้แปลว่าภาพปกติ ต้องวาดขอบเขตจากอาการที่เห็นจริง',
            'resolved_after_manual_fix':'ตรวจภาพซูมแล้วแก้กรอบให้ครอบรอยที่เห็นได้ครบ ชื่อโรคอ้างอิงโฟลเดอร์ต้นทาง ยังไม่ใช่การรับรองโดยผู้เชี่ยวชาญ',
        }
        if status=='crop_candidates_need_review':
            matching=[o for o in result['objects'] if o['class_id']==r['source_class_id']]
            if not matching:
                reasons[status]='เมื่อซูมมีกรอบเสนอขึ้นมา แต่ชื่อโรคที่โมเดลทายไม่ตรงกับชื่อโฟลเดอร์ จึงยังต้องตรวจชื่อโรคและขอบเขตจริงก่อนใช้'
            elif all(o.get('touches_internal_crop_edge') for o in matching):
                reasons[status]='พบกรอบที่ชื่อโรคตรงโฟลเดอร์ แต่ขอบเขตชนขอบภาพซูมและอาจถูกตัด จึงยังไม่ใช่ annotation ที่ล้อมครบ'
        note='AI ดูภาพจากแผ่นรวมภาพแล้ว; ไม่ใช่การรับรองชื่อโรคโดยผู้เชี่ยวชาญ'
        if r['sha256'] in audits and status!='resolved_after_manual_fix':
            code=audits[r['sha256']]['review_code']
            observations={
                'background_or_wrong_structure':'ตรวจแล้วมีกรอบชี้ไปที่พื้นหลัง มือ หรือส่วนที่ไม่ใช่รอยเป้าหมาย จึงยังใช้เป็น label ไม่ได้',
                'incomplete_instance_or_lesion_coverage':'ตรวจแล้วได้กรอบเพียงบางจุด รอยหรือวัตถุอื่นที่เห็นยังตกหล่น จึงยังล้อมไม่ครบ',
                'leaf_vs_symptom_boundary_not_confirmed':'กรอบครอบบางส่วนของใบ แต่ยังไม่ยืนยันว่าแนวกรอบตรงบริเวณอาการ จึงยังต้องวาดหรือตรวจแก้',
                'source_class_or_geometry_disagreement':'ชื่อโรคที่ทำนายหรือบริเวณที่ล้อมยังไม่ตรงกับข้อมูลต้นทาง ต้องตรวจชื่อโรคและขอบเขตก่อนใช้',
            }
            if code in observations:reasons[status]=observations[code]
        if q['small_source_resolution']:note+=' · ต้นฉบับมีขนาดเล็ก การขยายไม่เพิ่มรายละเอียดของรอยโรค'
        if 'source_folders_disagree_on_class' in r['reasons']:note+=' · รูปเดียวกันอยู่คนละชื่อโรคในโฟลเดอร์ต้นทาง'
        image='images/'+r['id']+'.jpg'
        if not (out/image).exists():copy_file(source/r['image'],out/image)
        candidate='candidates/'+r['id']+'.json'
        write_json(out/candidate,{**result,'source_class_id':r['source_class_id'],'expert_diagnosis_confirmed':False})
        manifest.append({'id':r['id'],'sha256':r['sha256'],'source':r['source'],
            'filename':Path(r['source']).name,'source_folder':r['source_folder'],
            'source_class_id':r['source_class_id'],'group':'new' if r['source_class_id']>=9 else 'old',
            'image':image,'candidate_file':candidate,'candidate_count':len(result['objects']),
            'status':status,'reason':reasons[status],'visual_note':note,
            'width':q['width'],'height':q['height'],'duplicate_count':counts[r['sha256']],
            'training_label_approved':r['sha256'] in approved})
    write_json(out/'review_manifest.json',manifest)
    fields=['id','filename','source_folder','group','status','reason','width','height','candidate_count','source']
    with (out/'remaining_status.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in manifest:w.writerow({k:r[k] for k in fields})
    shutil.copyfile(Path(__file__).parent/'rice_remaining_review.html',out/'review.html')
    summary=json.loads((out/'summary.json').read_text())
    summary['visually_checked_additions_files']=sum(r['training_label_approved'] for r in manifest)
    summary['remaining_after_manual_review']=sum(not r['training_label_approved'] for r in manifest)
    summary['pending_status_counts']=dict(Counter(r['status'] for r in manifest if not r['training_label_approved']))
    write_json(out/'summary.json',summary)
    assert len(manifest)==308 and sum(r['group']=='new' for r in manifest)==145
    assert all((out/r['image']).is_file() for r in manifest)
    print(json.dumps({'gallery_files':len(manifest),'pending_files':summary['remaining_after_manual_review'],
        'crop_candidate_files_not_approved':sum(r['status']=='crop_candidates_need_review' for r in manifest),
        'default_url':'http://127.0.0.1:8769/review.html'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('temp/rice_labeling/rice_full_auto'))
    p.add_argument('--output',type=Path,default=Path('temp/rice_labeling/remaining_recheck'))
    a=p.parse_args();publish(a.source,a.output)
