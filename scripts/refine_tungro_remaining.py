"""Refine unresolved images using individually selected visible-leaf points."""
from pathlib import Path
import json,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_label_rice import write_json,valid_polygon
from label_tungro import ROOT,WORK,overlay
from segment_tungro_fastsam import shape


def extract(rgb,masks,points):
    h,w=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    hue,sat,val=[hsv[:,:,i] for i in range(3)]
    visible=(hue>=3)&(hue<=40)&(sat>=28)&(val>=55)
    leaf=np.zeros((h,w),np.uint8)
    for mask in masks:
        if mask.shape!=(h,w):mask=cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        area=int(mask.sum())
        if area<5 or area>h*w*.5:continue
        if not any(mask[min(h-1,round(y)),min(w-1,round(x))] for x,y in points):continue
        ratio,solidity=shape(mask)
        if ratio<1.8 or solidity<.3:continue
        leaf|=mask.astype(np.uint8)
    selected=(leaf.astype(bool)&visible).astype(np.uint8)
    objects=[];n,labels,stats,_=cv2.connectedComponentsWithStats(selected,8)
    for k in range(1,n):
        region=labels==k;area=stats[k,cv2.CC_STAT_AREA]
        if area<max(4,w*h*.00002):continue
        ratio,_=shape(region)
        near=any(region[min(h-1,round(y)),min(w-1,round(x))] for x,y in points)
        if ratio<2 and not near:continue
        contours,_=cv2.findContours(region.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        c=max(contours,key=cv2.contourArea)
        if cv2.contourArea(c)<3:continue
        c=cv2.approxPolyDP(c,.35,True).reshape(-1,2);p=valid_polygon(c/[w,h])
        if p:objects.append(dict(class_id=9,polygon=p,method='visually_selected_leaf_point_SAM_and_visible_discoloration',diagnosis_confirmed=False))
    return objects


def run():
    import torch
    from ultralytics import SAM
    torch.set_num_threads(4);cv2.setNumThreads(1)
    rows=json.loads((WORK/'remaining_selection.json').read_text())
    points=json.loads((WORK/'manual_points_224.json').read_text())
    dest=WORK/'manual_sam';dest.mkdir(exist_ok=True)
    model=SAM('temp/rice_labeling/models/sam2.1_t.pt');start=time.monotonic()
    for n,pts224 in points.items():
        r=rows[int(n)-1];path=dest/(r['id']+'.json')
        if path.exists() and json.loads(path.read_text()).get('schema')==2:continue
        with Image.open(ROOT/r['image']) as im:
            im=im.convert('RGB');im.thumbnail((960,960));rgb=np.asarray(im)
        h,w=rgb.shape[:2];points_native=[]
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).astype(float)
        hue,sat,val=[hsv[:,:,i] for i in range(3)]
        score=(sat/255)*(val/255)**2*np.exp(-((hue-23)/12)**2)
        score[(hue<3)|(hue>40)|(sat<25)|(val<55)]=0
        score=cv2.GaussianBlur(score,(3,3),0)
        for x,y in pts224:
            cx,cy=x/224*w,y/224*h;rx,ry=12/224*w,12/224*h
            x0,y0=max(0,int(cx-rx)),max(0,int(cy-ry));x1,y1=min(w,int(cx+rx+1)),min(h,int(cy+ry+1))
            yy,xx=np.mgrid[y0:y1,x0:x1]
            local=score[y0:y1,x0:x1]*np.exp(-(((xx-cx)/max(rx,1))**2+((yy-cy)/max(ry,1))**2)/2)
            sy,sx=np.unravel_index(local.argmax(),local.shape)
            points_native.append([int(x0+sx),int(y0+sy)])
        masks=[];scores=[]
        if points_native:
            result=model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),points=[[[x,y]] for x,y in points_native],
                labels=[[1]]*len(points_native),conf=.05,device='mps',verbose=False)[0]
            masks=result.masks.data.cpu().numpy()>0 if result.masks is not None else []
            scores=result.boxes.conf.tolist()
        objects=extract(rgb,masks,points_native)
        write_json(path,dict(id=r['id'],schema=2,remaining_index=int(n),objects=objects,points_224=pts224,
            snapped_points_224=[[x/w*224,y/h*224] for x,y in points_native],segmentation_quality_scores_not_disease_confidence=scores,
            prompt_source='Codex individual visual selection on source image',visually_accepted=False,
            expert_diagnosis_confirmed=False,semantic_class_source='unconfirmed_supplied_Tungro_category'))
        print(json.dumps(dict(stage='manual_point_refinement',index=int(n),total=len(points),objects=len(objects),elapsed=round(time.monotonic()-start))),flush=True)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
    indices=[int(x) for x in points]
    out=WORK/'manual_remaining_preview';out.mkdir(exist_ok=True)
    for start in range(0,len(indices),12):
        page=Image.new('RGB',(1344,1056),'white');d=ImageDraw.Draw(page)
        for j,n in enumerate(indices[start:start+12]):
            r=rows[n-1];data=json.loads((dest/(r['id']+'.json')).read_text());x=j%3*448;y=j//3*264
            with Image.open(ROOT/r['image']) as im:
                before=im.convert('RGB');before.thumbnail((216,224));after=overlay(im,data['objects']);after.thumbnail((216,224))
            page.paste(before,(x,y+28));page.paste(after,(x+224,y+28));d.text((x+3,y+3),str(n)+' | '+str(len(data['objects']))+' regions',font=font,fill='black')
        page.save(out/('page_%02d.jpg'%(start//12+1)),quality=95)


if __name__=='__main__':run()
