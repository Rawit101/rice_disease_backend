"""Object-boundary support for Tungro drafts; this model cannot diagnose Tungro."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_label_rice import valid_polygon,write_json
from label_tungro import ROOT,WORK,overlay


def shape(mask):
    yy,xx=np.where(mask)
    if len(xx)<5:return 0,0
    eig=np.linalg.eigvalsh(np.cov(np.array([xx,yy])))
    elongation=float(np.sqrt(max(eig[-1],0)/max(eig[0],.25)))
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    area=sum(cv2.contourArea(c) for c in contours)
    hull=cv2.convexHull(np.vstack(contours))
    return elongation,area/max(1,cv2.contourArea(hull))


def extract(rgb,masks):
    h,w=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hue,sat,val=[hsv[:,:,i] for i in range(3)]
    plant=(hue>=5)&(hue<=95)&(sat>55)&(val>40)
    core=(hue>=5)&(hue<=32)&(sat>=100)&(val>=85)
    support=(hue>=3)&(hue<=36)&(sat>=65)&(val>=45)
    output=[];accepted_leaf_masks=[]
    for mask in masks:
        if mask.shape!=(h,w):mask=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        area=int(mask.sum())
        if area<8 or area>h*w*.5:continue
        elong,solid=shape(mask)
        if elong<2.6 or solid<.4 or np.count_nonzero(mask&plant)/area<.65:continue
        if np.count_nonzero(mask&core)<max(4,area*.035):continue
        accepted_leaf_masks.append(mask)
        selected=(mask&support).astype(np.uint8)
        selected=cv2.morphologyEx(np.pad(selected,2),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))[2:-2,2:-2]&mask.astype(np.uint8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(selected,8)
        for k in range(1,n):
            region=labels==k;size=stats[k,cv2.CC_STAT_AREA]
            if size<max(5,w*h*.00004) or np.count_nonzero(region&core)<max(3,size*.10):continue
            ratio,solidity=shape(region)
            if ratio<2.3:continue
            cs,_=cv2.findContours(region.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            c=max(cs,key=cv2.contourArea)
            c=cv2.approxPolyDP(c,max(.3,cv2.arcLength(c,True)*.0008),True).reshape(-1,2)
            polygon=valid_polygon(c/[w,h])
            if not polygon:continue
            output.append((region,dict(class_id=9,polygon=polygon,
                method='class_agnostic_leaf_mask_with_visible_symptom_tissue',
                diagnosis_confirmed=False,area_fraction=size/(w*h))))
    output.sort(key=lambda pair:pair[1]['area_fraction'],reverse=True)
    kept=[]
    for region,obj in output:
        if any(np.count_nonzero(region&old)/max(1,min(region.sum(),old.sum()))>.65 for old,_ in kept):continue
        kept.append((region,obj))
    return [o for _,o in kept],len(accepted_leaf_masks)


def run(all_images=False):
    from ultralytics import FastSAM
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)
    rows=json.loads((WORK/('inventory.json' if all_images else 'sample_selection.json')).read_text())
    dest=WORK/'fastsam';dest.mkdir(exist_ok=True)
    model=FastSAM('temp/rice_labeling/models/FastSAM-s.pt')
    start=time.monotonic()
    for i,r in enumerate(rows,1):
        path=dest/(r['id']+'.json')
        if path.exists():continue
        with Image.open(ROOT/r['image']) as im:
            im=im.convert('RGB');im.thumbnail((960,960));rgb=np.asarray(im)
        result=model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),imgsz=1024,device='mps',
            conf=.2,iou=.85,retina_masks=True,verbose=False)[0]
        masks=result.masks.data.cpu().numpy()>0 if result.masks is not None else []
        objects,n=extract(rgb,masks)
        write_json(path,dict(id=r['id'],objects=objects,class_agnostic_leaf_masks=n,
            method='FastSAM-s_object_masks_and_visible_symptom_region',
            diagnosis_from_model=False,semantic_class_source='unconfirmed_Tungro_folder',
            visually_reviewed=False,expert_diagnosis_confirmed=False))
        print(json.dumps(dict(stage='tungro_object_support',done=i,total=len(rows),objects=len(objects),elapsed=round(time.monotonic()-start))),flush=True)
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
            page.save(WORK/('fastsam_sample_%02d.jpg'%(start//16+1)),quality=94)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');a=p.parse_args();run(a.all)
