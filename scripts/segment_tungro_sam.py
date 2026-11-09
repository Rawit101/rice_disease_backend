"""Point-prompted leaf segmentation, with symptom-only region proposals."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_label_rice import write_json
from label_tungro import ROOT,WORK,overlay
from segment_tungro_fastsam import extract


def seeds(rgb):
    h,w=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hue,sat,val=[hsv[:,:,i].astype(float) for i in range(3)]
    score=(sat/255)*(val/255)*np.exp(-((hue-23)/12)**2)
    score[(hue<3)|(hue>35)|(sat<70)|(val<70)]=0
    score[:max(1,h//50)]=0;score[-max(1,h//50):]=0
    score[:,:max(1,w//50)]=0;score[:,-max(1,w//50):]=0
    score=cv2.GaussianBlur(score,(3,3),0)
    out=[]
    for _ in range(4):
        y,x=np.unravel_index(np.argmax(score),score.shape)
        if score[y,x]<.14:break
        out.append([int(x),int(y)])
        cv2.circle(score,(int(x),int(y)),max(16,min(w,h)//5),0,-1)
    return out


def run(all_images=False,variant='b',limit=None):
    from ultralytics import SAM
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    rows=json.loads((WORK/('inventory.json' if all_images else 'sample_selection.json')).read_text())
    if all_images:
        rows.sort(key=lambda r:(not (r['width']==224 and r['height']==224),r['source']))
    if limit:rows=rows[:limit]
    suffix='' if variant=='b' else '_'+variant
    dest=WORK/('sam'+suffix);dest.mkdir(exist_ok=True)
    model=SAM('temp/rice_labeling/models/sam2.1_'+variant+'.pt');start=time.monotonic()
    for i,r in enumerate(rows,1):
        path=dest/(r['id']+'.json')
        if path.exists():continue
        with Image.open(ROOT/r['image']) as im:
            im=im.convert('RGB');im.thumbnail((960,960));rgb=np.asarray(im)
        pts=seeds(rgb);objects=[];n=0
        if pts:
            result=model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),points=[[[x,y]] for x,y in pts],
                labels=[[1]]*len(pts),device='mps',verbose=False)[0]
            masks=result.masks.data.cpu().numpy()>0 if result.masks is not None else []
            objects,n=extract(rgb,masks)
        write_json(path,dict(id=r['id'],objects=objects,points=pts,class_agnostic_leaf_masks=n,
            method='SAM2.1_'+variant+'_positive_point_leaf_support_and_symptom_region',diagnosis_from_model=False,
            semantic_class_source='unconfirmed_Tungro_folder',visually_reviewed=False,expert_diagnosis_confirmed=False))
        print(json.dumps(dict(stage='tungro_sam_support',done=i,total=len(rows),objects=len(objects),elapsed=round(time.monotonic()-start))),flush=True)
    if not all_images:
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
        for start in range(0,len(rows),16):
            page=Image.new('RGB',(1344,1056),'white');d=ImageDraw.Draw(page)
            for j,r in enumerate(rows[start:start+16]):
                data=json.loads((dest/(r['id']+'.json')).read_text())
                with Image.open(ROOT/r['image']) as im:
                    before=im.convert('RGB');before.thumbnail((164,224));after=overlay(im,data['objects']);after.thumbnail((164,224))
                x=j%4*336;y=j//4*264;page.paste(before,(x,y+24));page.paste(after,(x+168,y+24))
                d.text((x+2,y+3),f"{start+j+1:03d} | {len(data['objects'])} masks",font=font,fill='black')
            page.save(WORK/('sam'+suffix+'_sample_%02d.jpg'%(start//16+1)),quality=94)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true')
    p.add_argument('--variant',choices=['b','t'],default='b');p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.all,a.variant,a.limit)
