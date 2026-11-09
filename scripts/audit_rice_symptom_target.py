"""Flag green-dominated masks for review under the owner's symptom-only policy.

This heuristic never diagnoses a disease, edits a polygon, or declares a mask bad.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import json
import cv2, numpy as np
from PIL import Image
from audit_rice_10class import BASE, OUT, read, write
from publish_rice_class_names import parse_label, NAMES


def one(r):
    if not r['object_count'] or not set(r['class_ids']).intersection({0,1,2,3,8,9}):
        return r
    root=BASE/r['dataset_root'];preview=root/r.get('thumbnail','missing')
    if not preview.is_file():preview=root/r['image']
    with Image.open(preview) as original:
        image=original.convert('RGB');image.thumbnail((480,480));rgb=np.asarray(image)
    objects=parse_label((root/r['label']).read_text(),NAMES)
    mask=np.zeros(rgb.shape[:2],np.uint8)
    for o in objects:
        if o['class_id'] not in {0,1,2,3,8,9}:continue
        cv2.fillPoly(mask,[np.rint(np.asarray(o['polygon'])*[image.width-1,image.height-1]).astype(np.int32)],1)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    green=(hsv[:,:,0]>=35)&(hsv[:,:,0]<=90)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)
    fraction=float(np.count_nonzero(green&(mask>0))/max(1,mask.sum()))
    r['green_fraction_under_leaf_disease_masks']=fraction
    if fraction>.65 and mask.mean()>.01:
        r['audit_flags']=sorted(set(r['audit_flags'])|{'green_tissue_dominates_mask_requires_symptom_boundary_review'})
    return r


def run():
    cv2.setNumThreads(1);rows=read(OUT/'audit_manifest.json');new=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(one,rows),1):
            new.append(r)
            if i%2000==0:print(json.dumps(dict(stage='symptom_target_check',completed=i,total=len(rows))),flush=True)
    write(OUT/'audit_manifest.json',new)
    report=read(OUT/'audit_report.json');report['flags']=dict(Counter(f for r in new for f in r['audit_flags']))
    report['target_policy']='visible_symptom_regions_only'
    report['green_mask_check_is_a_review_signal_not_diagnosis']=True
    write(OUT/'audit_report.json',report)
    print(json.dumps(dict(stage='symptom_target_check_complete',green_dominated_masks=report['flags'].get('green_tissue_dominates_mask_requires_symptom_boundary_review',0))),flush=True)


if __name__=='__main__':run()
