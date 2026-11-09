"""Manual visible-leaf seeds with local color selection; not disease inference."""
import json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_label_rice import valid_polygon,write_json
from label_tungro import ROOT,WORK,overlay


def run():
    rows=json.loads((WORK/'remaining_selection.json').read_text());points=json.loads((WORK/'manual_points_224.json').read_text())
    dest=WORK/'manual_seeded';dest.mkdir(exist_ok=True)
    for n,pts224 in points.items():
        r=rows[int(n)-1]
        with Image.open(ROOT/r['image']) as im:
            im=im.convert('RGB');im.thumbnail((960,960));rgb=np.asarray(im)
        h,w=rgb.shape[:2];hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV).astype(float);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(float)
        hue,sat,val=[hsv[:,:,i] for i in range(3)];score=(sat/255)*(val/255)**2*np.exp(-((hue-24)/12)**2)
        score[(hue<3)|(hue>41)|(sat<28)|(val<65)]=0;score=cv2.GaussianBlur(score,(3,3),0)
        chosen=np.zeros((h,w),np.uint8);snapped=[]
        for x,y in pts224:
            cx,cy=x/224*w,y/224*h;rx,ry=12/224*w,12/224*h
            x0,y0=max(0,int(cx-rx)),max(0,int(cy-ry));x1,y1=min(w,int(cx+rx+1)),min(h,int(cy+ry+1))
            yy,xx=np.mgrid[y0:y1,x0:x1];local=score[y0:y1,x0:x1]*np.exp(-(((xx-cx)/max(rx,1))**2+((yy-cy)/max(ry,1))**2)/2)
            sy,sx=np.unravel_index(local.argmax(),local.shape);sx+=x0;sy+=y0
            if score[sy,sx]<.025:continue
            snapped.append([sx/w*224,sy/h*224]);sh,ss,sv=hsv[sy,sx]
            distance=np.linalg.norm(lab-lab[sy,sx],axis=2)
            support=((hue>=max(3,sh-12))&(hue<=min(41,sh+12))&(sat>=max(28,ss*.6))&(val>=max(65,sv*.7))&(distance<50)).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(support,8);key=labels[sy,sx]
            if key and stats[key,cv2.CC_STAT_AREA]<h*w*.3:chosen[labels==key]=1
        objects=[];contours,_=cv2.findContours(chosen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c)<3:continue
            c=cv2.approxPolyDP(c,.35,True).reshape(-1,2);p=valid_polygon(c/[w,h])
            if p:objects.append(dict(class_id=9,polygon=p,method='individual_visible_leaf_seed_and_local_color_boundary',diagnosis_confirmed=False))
        write_json(dest/(r['id']+'.json'),dict(id=r['id'],remaining_index=int(n),objects=objects,manual_points_224=pts224,
            snapped_points_224=snapped,expert_diagnosis_confirmed=False,visually_accepted=False,
            semantic_class_source='unconfirmed_supplied_Tungro_category'))
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16);indices=[int(x) for x in points]
    out=WORK/'seeded_remaining_preview';out.mkdir(exist_ok=True)
    for start in range(0,len(indices),12):
        page=Image.new('RGB',(1344,1056),'white');d=ImageDraw.Draw(page)
        for j,n in enumerate(indices[start:start+12]):
            r=rows[n-1];data=json.loads((dest/(r['id']+'.json')).read_text());x=j%3*448;y=j//3*264
            with Image.open(ROOT/r['image']) as im:
                before=im.convert('RGB');before.thumbnail((216,224));after=overlay(im,data['objects']);after.thumbnail((216,224))
            page.paste(before,(x,y+28));page.paste(after,(x+224,y+28));d.text((x+3,y+3),str(n)+' | '+str(len(data['objects']))+' regions',font=font,fill='black')
        page.save(out/('page_%02d.jpg'%(start//12+1)),quality=95)
    print('manual visible-leaf seeds processed:',len(indices),flush=True)


if __name__=='__main__':cv2.setNumThreads(1);run()
