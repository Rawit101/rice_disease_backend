"""Generate pixel-boundary proposals. Never invent or rename a disease class.

Small trims are candidate results, not evidence of correct disease boundaries.
The publisher holds all color-only changes unless explicitly visually accepted.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json,time
import cv2,numpy as np
from PIL import Image
from auto_label_rice import valid_polygon,write_json


def audit_object(obj,hsv,w,h):
    xy=np.rint(np.asarray(obj['polygon'])*[w,h]).astype(np.int32)
    xy=np.clip(xy,[0,0],[w-1,h-1])
    x0,y0=xy.min(0);x1,y1=xy.max(0)+1
    crop=hsv[y0:y1,x0:x1]
    mask=np.zeros(crop.shape[:2],np.uint8);cv2.fillPoly(mask,[xy-[x0,y0]],1)
    hue,sat,val=[crop[:,:,i] for i in range(3)]
    symptom=(((hue<37)|(hue>165))&(sat>30)&(val>24))|((sat<75)&(val<180)&(val>20))|(val<50)
    visible=mask&symptom.astype(np.uint8)
    # Explicit padding prevents closing from extending a lesion into white
    # pixels at the tight ROI boundary through OpenCV's default border mode.
    visible=cv2.morphologyEx(np.pad(visible,2),cv2.MORPH_CLOSE,
        np.ones((3,3),np.uint8))[2:-2,2:-2]&mask
    count,labels,stats,_=cv2.connectedComponentsWithStats(visible,8)
    proposals=[];union=np.zeros(mask.shape,np.uint8)
    for k in range(1,count):
        if stats[k,cv2.CC_STAT_AREA]<3:continue
        comp=(labels==k).astype(np.uint8)
        contours,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour)<2:continue
        poly=cv2.approxPolyDP(contour,.2,True).reshape(-1,2)+[x0,y0]
        normalized=valid_polygon(poly/[w,h])
        if normalized:
            proposals.append(dict(obj,polygon=normalized))
            cv2.fillPoly(union,[np.rint(np.asarray(normalized)*[w,h]).astype(np.int32)-[x0,y0]],1)
    intersection=np.count_nonzero(union&mask);area=np.count_nonzero(union|mask)
    iou=intersection/max(1,area)
    fraction=float(visible.sum()/max(1,mask.sum()))
    safe=.9<=iou<.995 and bool(proposals)
    return dict(objects=proposals if safe else [obj],proposal_objects=proposals,
        repair_applied=safe,old_new_mask_iou=iou,visible_discoloration_fraction=fraction,
        needs_visual_boundary_review=iou<.9,
        no_visible_discoloration_under_mask=fraction<.03)


def run():
    root=Path('temp/rice_labeling/rice_archive_auto');out=Path('temp/rice_labeling/archive_corrections')
    dest=out/'boundary';dest.mkdir(exist_ok=True)
    records=json.loads((root/'manifest.json').read_text())
    canonical={r['sha256']:r for r in reversed(records) if r['label']}
    stats=Counter();start=time.monotonic()
    def one(item):
        sha,r=item;path=dest/(sha+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('schema')==3:return saved
        geo=json.loads((out/'geometry'/(sha+'.json')).read_text())
        with Image.open(root/r['image']) as source:
            im=source.convert('RGB');im.thumbnail((1600,1600));rgb=np.asarray(im)
        h,w=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        results=[audit_object(o,hsv,w,h) for o in geo['objects']]
        obj=[o for x in results for o in x['objects']]
        value=dict(schema=3,id=r['id'],sha256=sha,objects=obj,object_audit=results,
            boundary_changes=sum(x['repair_applied'] for x in results),
            geometry_changes=geo['changed_objects'],semantic_classes_unchanged=True,
            visually_reviewed=False)
        write_json(path,value);return value
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,value in enumerate(pool.map(one,canonical.items()),1):
            stats['unique_labeled_contents_checked']+=1
            stats['unique_contents_with_safe_boundary_trim']+=bool(value['boundary_changes'])
            stats['objects_trimmed']+=value['boundary_changes']
            stats['contents_needing_visual_boundary_review']+=any(x['needs_visual_boundary_review'] for x in value['object_audit'])
            if i%500==0:print(json.dumps(dict(stage='image_boundary_audit',done=i,total=len(canonical),elapsed=round(time.monotonic()-start),**stats)),flush=True)
    write_json(out/'boundary_summary.json',dict(stats));print(json.dumps(dict(stage='complete',**stats)),flush=True)


if __name__=='__main__':cv2.setNumThreads(1);run()
