"""Repair polygon topology and prepare visually adjudicated boundary corrections.

Geometry cleanup does not validate a disease. Folder hints never become a new
semantic label without an explicit visual decision. Original exports are intact.
"""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse,json,math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from auto_label_rice import ORIGINAL_NAMES,valid_polygon,write_json,yolo_text
from build_full_rice_comparison import merge_predictions


def cycles(points):
    """Split a closed contour walk at repeated vertices; discard zero-area bridges."""
    stack=[];positions={};result=[]
    for point in list(points)+[points[0]]:
        key=tuple(round(float(v),9) for v in point)
        if key in positions:
            start=positions[key]
            polygon=valid_polygon(stack[start:])
            if polygon:result.append(polygon)
            for old in stack[start+1:]:positions.pop(tuple(round(float(v),9) for v in old),None)
            stack=stack[:start+1]
        else:
            positions[key]=len(stack);stack.append(list(point))
    return result or [points]


def geometry(objects):
    fixed=[];changes=0
    for i,o in enumerate(objects):
        parts=cycles(o['polygon'])
        areas=[sum(p[j][0]*p[(j+1)%len(p)][1]-p[j][1]*p[(j+1)%len(p)][0]
                   for j in range(len(p))) for p in parts]
        # Opposite winding can describe a hole rather than separate objects.
        # Keep the input until a reviewer adjudicates it instead of filling it.
        if any(a>0 for a in areas) and any(a<0 for a in areas):
            fixed.append(dict(o,parent_object_index=i,topology_requires_review=True))
            continue
        changed=len(parts)!=1 or parts[0]!=o['polygon']
        changes+=changed
        for polygon in parts:
            fixed.append(dict(o,polygon=polygon,parent_object_index=i,
                              topology_bridge_removed=changed))
    return fixed,changes


def refine(rgb,obj,cls):
    h,w=rgb.shape[:2];p=np.asarray(obj['polygon'],np.float32)*[w,h]
    a=p.min(0);b=p.max(0);margin=np.maximum(4,(b-a)*.25)
    x0,y0=np.maximum(0,np.floor(a-margin)).astype(int)
    x1,y1=np.minimum([w,h],np.ceil(b+margin)).astype(int)
    if x1-x0<5 or y1-y0<5:return []
    crop=rgb[y0:y1,x0:x1];oh,ow=crop.shape[:2]
    scale=min(1.,640/max(ow,oh));cw,ch=max(1,round(ow*scale)),max(1,round(oh*scale))
    crop=cv2.resize(crop,(cw,ch));xy=np.rint((p-[x0,y0])*[cw/ow,ch/oh]).astype(np.int32)
    seed=np.zeros((ch,cw),np.uint8);cv2.fillPoly(seed,[xy],1)
    if seed.sum()<3 or seed.mean()>.92:return []
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    symptom=((hsv[:,:,0]<36)&(hsv[:,:,1]>25)&(hsv[:,:,2]>18))|((hsv[:,:,2]<110)&(hsv[:,:,1]<100))
    inside=cv2.erode(seed,np.ones((3,3),np.uint8))
    foreground=inside.astype(bool) if cls==4 else inside.astype(bool)&symptom
    if foreground.sum()<2:foreground=inside.astype(bool)
    if foreground.sum()<2:return []
    gc=np.full(seed.shape,cv2.GC_PR_BGD,np.uint8);gc[seed>0]=cv2.GC_PR_FGD
    gc[foreground]=cv2.GC_FGD
    far=cv2.dilate(seed,np.ones((7,7),np.uint8))==0
    gc[far]=cv2.GC_BGD
    if not far.any():return []
    try:
        cv2.grabCut(cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),gc,None,
            np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
    except cv2.error:return []
    mask=((gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD)).astype(np.uint8)
    if cls in [6,7]:
        mask&=symptom.astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,components,stats,_=cv2.connectedComponentsWithStats(mask,8)
    output=[]
    for k in range(1,count):
        part=(components==k).astype(np.uint8)
        if stats[k,4]<4 or np.count_nonzero(part&seed)<2:continue
        contours,_=cv2.findContours(part,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour)<3:continue
        contour=cv2.approxPolyDP(contour,.35,True).reshape(-1,2)
        normalized=(contour*[ow/cw,oh/ch]+[x0,y0])/[w,h]
        polygon=valid_polygon(normalized)
        if polygon:output.append(dict(class_id=cls,polygon=polygon,
            method='grabcut_seeded_boundary_and_visible_discoloration_proposal',
            source_seed_class_id=obj['class_id']))
    return output


def overlay(image,objects):
    im=image.copy();im.thumbnail((224,224));layer=Image.new('RGBA',im.size);draw=ImageDraw.Draw(layer)
    for obj in objects:
        pts=[(x*im.width,y*im.height) for x,y in obj['polygon']]
        draw.polygon(pts,fill=(255,40,70,45),outline=(255,10,55,255),width=1)
    return Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')


def prepare(args):
    root,out=args.source,args.output
    for sub in ['geometry','proposals','sheets','individual']:(out/sub).mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/'manifest.json').read_text());by=defaultdict(list)
    for r in rows:by[r['sha256']].append(r)
    eligible={sha:next((r for r in rs if r['source_class_id'] is not None and r['source_class_id']<9),rs[0])
              for sha,rs in by.items()}
    stats=Counter()
    for i,(sha,r) in enumerate(eligible.items(),1):
        if not r['label']:continue
        dest=out/'geometry'/(sha+'.json')
        record=json.loads(dest.read_text()) if dest.exists() else None
        if not record or record.get('schema')!=2:
            before=json.loads((root/r['prediction_file']).read_text())['objects']
            after,changed=geometry(before)
            record=dict(schema=2,sha256=sha,id=r['id'],objects=after,changed_objects=changed,
                original_objects=len(before),semantic_classes_unchanged=True,visually_reviewed=False)
            write_json(dest,record)
        stats['unique_labeled_contents_checked']+=1
        if record['changed_objects']:
            stats['unique_contents_with_topology_repair']+=1
            stats['source_files_with_topology_repair']+=sum(bool(x['label']) for x in by[sha])
        if i%1000==0:print(json.dumps(dict(stage='polygon_audit',processed=i,**stats)),flush=True)
    write_json(out/'geometry_summary.json',dict(stats))
    small=[r for r in eligible.values() if r['source_class_id'] in [4,6,7]]
    small.sort(key=lambda r:(r['source_class_id'],r['source']))
    write_json(out/'visual_selection.json',small)
    def proposal(r):
        dest=out/'proposals'/(r['sha256']+'.json')
        if dest.exists():return
        data=json.loads((root/r['prediction_file']).read_text())
        expected=r['source_class_id']
        candidates=[o for key,objects in data['raw_models'].items() for o in objects if o['class_id']==expected]
        if not candidates:candidates=data['objects']
        # Class changes here are explicitly unconfirmed proposals. Only visual
        # adjudication may turn one into an annotation for the hinted class.
        candidates,_=merge_predictions([('seed',candidates)])
        with Image.open(root/r['image']) as im:
            rgb=np.asarray(im.convert('RGB'));objects=[]
            for seed in candidates[:40]:objects.extend(refine(rgb,seed,expected))
        write_json(dest,dict(id=r['id'],sha256=r['sha256'],objects=objects,
            semantic_class_is_folder_hint_only=True,is_training_label=False,
            before=data['objects'],no_seed=not candidates))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,_ in enumerate(pool.map(proposal,small),1):
            if i%25==0:print(json.dumps(dict(stage='boundary_proposals',completed=i,total=len(small))),flush=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    for start in range(0,len(small),12):
        page=Image.new('RGB',(1440,1120),'white');draw=ImageDraw.Draw(page)
        for j,r in enumerate(small[start:start+12]):
            data=json.loads((out/'proposals'/(r['sha256']+'.json')).read_text())
            x=j%3*480;y=j//3*280
            with Image.open(root/r['image']) as original:
                a=original.convert('RGB');a.thumbnail((224,224));b=overlay(original.convert('RGB'),data['objects'])
                page.paste(a,(x+(240-a.width)//2,y+24));page.paste(b,(x+240+(240-b.width)//2,y+24))
            draw.text((x+3,y+3),'%03d %s | candidate %d'%(start+j+1,r['source_folder'],len(data['objects'])),font=font,fill='black')
            draw.text((x+3,y+252),Path(r['source']).name[:55],font=font,fill='black')
        page.save(out/'sheets'/('page_%02d.jpg'%(start//12+1)),quality=95)
    print(json.dumps(dict(stage='prepared',visual_images=len(small),**stats)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('temp/rice_labeling/rice_archive_auto'))
    p.add_argument('--output',type=Path,default=Path('temp/rice_labeling/archive_corrections'))
    args=p.parse_args();cv2.setNumThreads(1);prepare(args)
