"""Screen image quality separately from disease identity and annotation correctness.

Metrics identify risks, not ground-truth quality. Only individually inspected
images are promoted into the selected photo set by the publication step.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, math, random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from audit_rice_10class import BASE, read, write

OUT=BASE/'rice_image_quality'


def border_repeat_fraction(rgb):
    """Flag edge replication from image warping, not ordinary plain backgrounds."""
    h,w=rgb.shape[:2];bands=[]
    for axis in [0,1]:
        a=rgb if axis==0 else np.transpose(rgb,(1,0,2))
        # Plain white/black cutout backgrounds do not establish synthetic borders.
        if float(a[0].std())<12: first=0
        else:
            first=0
            for j in range(1,min(len(a)//3,160)):
                if float(np.abs(a[j].astype(float)-a[0].astype(float)).mean())>2.5:break
                first=j
        if float(a[-1].std())<12:last=0
        else:
            last=0
            for j in range(1,min(len(a)//3,160)):
                if float(np.abs(a[-j-1].astype(float)-a[-1].astype(float)).mean())>2.5:break
                last=j
        bands.extend([first/max(1,len(a)),last/max(1,len(a))])
    return max(bands)


def measure(r):
    p=BASE/r['dataset_root']/r['image'];risks=[]
    result=dict(id=r['id'],source_sha256=r['sha256'],source_folder=r['source_folder'],
                expected_class_id=r['expected_class_id'],dataset_root=r['dataset_root'],
                image=r['image'],filename=Path(r['source']).name,
                label_quality_does_not_decide_photo_quality=True,
                disease_confirmed=False,visually_inspected_for_photo_selection=False)
    try:
        with Image.open(p) as original:
            im=original.convert('RGB');w,h=im.size
            # Exact decoded-pixel duplicate identity survives filename/JPEG-container differences.
            digest=hashlib.sha256(str((w,h)).encode()+im.tobytes()).hexdigest()
            im.thumbnail((480,480));rgb=np.asarray(im)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        nonblank=(hsv[:,:,2]>12)&~((hsv[:,:,1]<12)&(hsv[:,:,2]>245))
        interior=cv2.erode(nonblank.astype(np.uint8),np.ones((3,3),np.uint8))>0
        lap=cv2.Laplacian(gray,cv2.CV_32F)
        focus=float(lap[interior].var()) if interior.any() else 0.
        values=gray[nonblank] if nonblank.any() else gray.ravel()
        hist=np.bincount(values.ravel(),minlength=256).astype(float);hist/=max(1,hist.sum())
        entropy=float(-(hist[hist>0]*np.log2(hist[hist>0])).sum())
        contrast=float(np.percentile(values,95)-np.percentile(values,5))
        informative=float(nonblank.mean());replicated=border_repeat_fraction(rgb)
        if max(w,h)<160 or w*h<12000:risks.append('very_small_source_image')
        if informative<.005 or entropy<1.5:risks.append('nearly_blank_or_little_visible_content')
        if float((gray<12).mean())>.985:risks.append('nearly_all_black')
        if contrast<20:risks.append('low_visible_contrast')
        if focus<15:risks.append('low_focus_metric_needs_visual_check')
        if replicated>.12:risks.append('large_replicated_edge_band')
        if r.get('exact_content_source_categories') and len(r['exact_content_source_categories'])>1:
            risks.append('source_categories_conflict_not_an_image_quality_failure')
        # Ranking only. It never accepts an image or establishes a disease.
        rank=float(min(focus,700)/700+min(contrast,180)/180+min(entropy,7)/7-replicated*3)
        result.update(width=w,height=h,pixel_sha256=digest,focus_metric=focus,
                      visible_contrast=contrast,entropy=entropy,informative_fraction=informative,
                      replicated_border_fraction=replicated,ranking_score=rank,
                      metrics_review_flags=risks,image_decode_ok=True)
    except Exception as error:
        result.update(image_decode_ok=False,pixel_sha256=None,ranking_score=-100,
                      metrics_review_flags=['image_decode_error'],decode_error=str(error))
    return result


def scan():
    OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    rows=read(BASE/'rice_10class_audit/audit_manifest.json');results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,value in enumerate(pool.map(measure,rows),1):
            results.append(value)
            if i%1000==0:print(json.dumps(dict(stage='photo_quality_metrics',completed=i,total=len(rows))),flush=True)
    groups=defaultdict(list)
    for r in results:
        if r['pixel_sha256']:groups[r['pixel_sha256']].append(r)
    for group in groups.values():
        ids=sorted(r['id'] for r in group)
        for r in group:r.update(pixel_duplicate_group=ids[0],exact_pixel_duplicate_count=len(group))
    write(OUT/'quality_metrics.json',results)
    write(OUT/'metrics_summary.json',dict(source_files=len(results),exact_decoded_contents=len(groups),
        flags=dict(Counter(f for r in results for f in r['metrics_review_flags'])),
        automatic_quality_pass_is_not_visual_acceptance=True,training_labels_exported=0))
    print(json.dumps(dict(stage='photo_metrics_complete',files=len(results),decoded_contents=len(groups))),flush=True)


def sheets():
    rows=read(OUT/'quality_metrics.json');old=read(BASE/'rice_10class_audit/visual_selection.json')
    byid={r['id']:r for r in rows};chosen=[];seen=set()
    def take(r,reason):
        key=r.get('pixel_sha256') or r['id']
        if key in seen:return False
        seen.add(key);chosen.append(dict(r,selection_reason=reason));return True
    # Include the original audit photographs, but judge photo quality without masks.
    for entry in old:take(byid[entry['id']],'previous_annotation_audit_photo_recheck')
    rng=random.Random(20261103)
    for cls in range(10):
        pool=[r for r in rows if r['expected_class_id']==cls and r['image_decode_ok']]
        pool.sort(key=lambda r:(-r['ranking_score'],r['id']))
        # Review more potentially useful images in large classes, all rare-class contents.
        count=0
        for r in pool:
            if take(r,'additional_ranked_photo_candidate'):count+=1
            if count>=18:break
        risk=[r for r in pool if r['metrics_review_flags'] and
              any(f!='source_categories_conflict_not_an_image_quality_failure' for f in r['metrics_review_flags'])]
        rng.shuffle(risk);count=0
        for r in risk:
            if take(r,'quality_risk_example'):count+=1
            if count>=3:break
    chosen.sort(key=lambda r:(r['expected_class_id'],r['selection_reason'],r['id']))
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',13)
    folder=OUT/'contact_sheets';folder.mkdir(exist_ok=True)
    for start in range(0,len(chosen),12):
        page=Image.new('RGB',(1050,1400),'#e6e9e4');draw=ImageDraw.Draw(page)
        for j,r in enumerate(chosen[start:start+12]):
            index=start+j+1;x=j%3*350;y=j//3*350
            with Image.open(BASE/r['dataset_root']/r['image']) as original:
                image=original.convert('RGB');image.thumbnail((332,282))
            page.paste(image,(x+9+(332-image.width)//2,y+45+(282-image.height)//2))
            draw.text((x+7,y+3),'%03d hint=%d %s'%(index,r['expected_class_id'],r['id'][:14]),font=font,fill='black')
            draw.text((x+7,y+25),'%dx%d focus %.0f border %.2f'%(r['width'],r['height'],r['focus_metric'],r['replicated_border_fraction']),font=small,fill='black')
            r['visual_index']=index;r['contact_sheet']='contact_sheets/page_%02d.jpg'%(start//12+1)
        page.save(folder/('page_%02d.jpg'%(start//12+1)),quality=94)
    write(OUT/'visual_selection.json',chosen)
    print(json.dumps(dict(stage='photo_sheets_ready',individual_photos=len(chosen),pages=math.ceil(len(chosen)/12))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['scan','sheets']);a=p.parse_args()
    scan() if a.action=='scan' else sheets()
