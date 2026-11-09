"""Label visible yellow/orange leaf tissue as provisional Tungro annotations.

Image color supports the boundary proposal, not diagnosis. Tungro's identity
comes from the supplied category and remains explicitly unconfirmed.
"""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse,json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_label_rice import write_json,valid_polygon,yolo_text,ORIGINAL_NAMES

ROOT=Path('temp/rice_labeling/rice_archive_corrected')
WORK=Path('temp/rice_labeling/tungro_work')
CLASS_ID=9


def proposals(rgb):
    h,w=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hue,sat,val=[hsv[:,:,i] for i in range(3)]
    core=(hue>=7)&(hue<=32)&(sat>=140)&(val>=100)
    support=(hue>=4)&(hue<=33)&(sat>=110)&(val>=60)
    support=cv2.morphologyEx(np.pad(support.astype(np.uint8),2),cv2.MORPH_CLOSE,
        np.ones((3,3),np.uint8))[2:-2,2:-2]
    n,labels,stats,_=cv2.connectedComponentsWithStats(support,8)
    objects=[];held=[]
    for k in range(1,n):
        area=stats[k,cv2.CC_STAT_AREA]
        if area<max(6,h*w*.00008):continue
        component=labels==k;seed=int(np.count_nonzero(component&core))
        if seed<max(4,area*.25):continue
        yy,xx=np.where(component);cov=np.cov(np.array([xx,yy]))
        eig=np.linalg.eigvalsh(cov);elongation=float(np.sqrt(max(eig[-1],0)/max(eig[0],.25)))
        contours,_=cv2.findContours(component.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        c=max(contours,key=cv2.contourArea)
        if cv2.contourArea(c)<3:continue
        c=cv2.approxPolyDP(c,max(.35,cv2.arcLength(c,True)*.001),True).reshape(-1,2)
        poly=valid_polygon(c/[w,h])
        if not poly:continue
        value=dict(class_id=CLASS_ID,polygon=poly,method='visible_discolored_leaf_tissue_proposal',
            area_fraction=area/(w*h),elongation=elongation,core_fraction=seed/area,
            diagnosis_confirmed=False)
        if elongation>=3.5 and area/(w*h)<.25:objects.append(value)
        else:held.append(value)
    objects.sort(key=lambda o:o['area_fraction'],reverse=True)
    return objects[:24],held[:24]


def overlay(im,objects,size=224):
    im=im.copy();im.thumbnail((size,size));im=im.convert('RGBA')
    layer=Image.new('RGBA',im.size);d=ImageDraw.Draw(layer)
    for i,o in enumerate(objects):
        col=[(255,70,140),(20,220,240),(255,210,0),(130,90,255)][i%4]
        pts=[(x*im.width,y*im.height) for x,y in o['polygon']]
        d.polygon(pts,fill=col+(45,),outline=col+(255,),width=1)
    return Image.alpha_composite(im,layer).convert('RGB')


def run(sample=False):
    dest=WORK/'proposals';dest.mkdir(exist_ok=True,parents=True)
    rows=json.loads((WORK/('sample_selection.json' if sample else 'inventory.json')).read_text())
    def one(r):
        path=dest/(r['id']+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('schema')==2:return saved
        with Image.open(ROOT/r['image']) as original:
            im=original.convert('RGB');im.thumbnail((960,960));rgb=np.asarray(im)
        objects,held=proposals(rgb)
        value=dict(id=r['id'],schema=2,objects=objects,held_objects=held,
            semantic_class_source='unverified_supplied_Tungro_folder',
            expert_diagnosis_confirmed=False,visually_reviewed=False,
            inference_method='image_color_and_leaf_geometry_proposal_not_old_disease_model')
        write_json(path,value);return value
    stats=Counter();start=time.monotonic()
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,value in enumerate(pool.map(one,rows),1):
            stats['images_processed']+=1;stats['with_proposed_masks']+=bool(value['objects'])
            stats['mask_count']+=len(value['objects'])
            if i%100==0:print(json.dumps(dict(stage='tungro_boundary_proposals',elapsed=round(time.monotonic()-start),**stats)),flush=True)
    write_json(WORK/('sample_summary.json' if sample else 'proposal_summary.json'),dict(stats))
    print(json.dumps(dict(stage='complete',**stats)),flush=True)
    sheets(rows,WORK/('sample_proposals' if sample else 'review_sheets'))


def sheets(rows,dest,start_index=0):
    dest.mkdir(exist_ok=True,parents=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    for start in range(0,len(rows),16):
        page=Image.new('RGB',(1344,1056),'#f5f5ef');d=ImageDraw.Draw(page)
        for j,r in enumerate(rows[start:start+16]):
            data=json.loads((WORK/'proposals'/(r['id']+'.json')).read_text())
            with Image.open(ROOT/r['image']) as im:
                before=im.convert('RGB');before.thumbnail((164,224))
                after=overlay(im,data['objects'],224);after.thumbnail((164,224))
            x=j%4*336;y=j//4*264
            page.paste(before,(x,y+24));page.paste(after,(x+168,y+24))
            d.text((x+3,y+3),f"{start_index+start+j+1:04d} | {len(data['objects'])} masks | {Path(r['source']).name[:17]}",font=font,fill='black')
        page.save(dest/('page_%03d.jpg'%(start//16+1)),quality=94)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',action='store_true');a=p.parse_args();run(a.sample)
