"""Turn visually selected rice symptom ROIs into masks for final overlay review."""
import json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_label_rice import valid_polygon,write_json
from repair_rice_annotations import overlay

root=Path('temp/rice_labeling/rice_archive_auto')
out=Path('temp/rice_labeling/archive_corrections')
rows=json.loads((out/'visual_selection.json').read_text())
rois=json.loads((out/'manual_rois.json').read_text())
records=[]
for n,boxes in rois.items():
    r=rows[int(n)-1]
    with Image.open(root/r['image']) as source:im=source.convert('RGB')
    # ROI coordinates are selected by the reviewer on the 224px square images.
    if im.size!=(224,224):
        raise ValueError('Use normalized/manual full-size coordinates for '+r['id'])
    rgb=np.asarray(im);objects=[]
    for box_index,box in enumerate(boxes):
        x0,y0,x1,y1=map(int,box)
        x0,y0=max(0,x0),max(0,y0);x1,y1=min(224,x1),min(224,y1)
        cx0,cy0=x0,y0;crop=rgb[y0:y1,x0:x1]
        hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
        if int(n)==27 and box_index==0:
            mask=((hsv[:,:,0]<35)&(hsv[:,:,1]>90)&(hsv[:,:,2]>65)).astype(np.uint8)
        else:
            threshold,_=cv2.threshold(hsv[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask=((hsv[:,:,2]<min(155,threshold+15))&
                ((hsv[:,:,0]<40)|(hsv[:,:,1]<90)|(hsv[:,:,2]<65))).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labs,stats,_=cv2.connectedComponentsWithStats(mask,8)
        if count<2:raise ValueError('Manual ROI has no visible symptom component')
        component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        mask=(labs==component).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        p=cv2.approxPolyDP(max(contours,key=cv2.contourArea),.4,True).reshape(-1,2)
        p=valid_polygon((p+[cx0,cy0])/224)
        if not p:raise ValueError('Empty manual foreground ROI')
        objects.append(dict(class_id=r['source_class_id'],polygon=p))
    result=dict(id=r['id'],sha256=r['sha256'],audit_index=int(n),objects=objects,
        method='visually_selected_symptom_rois_color_contour_refinement',reviewer='Codex',
        expert_diagnosis_confirmed=False,visually_accepted=False,
        classification_basis='source category and visible symptom morphology')
    write_json(out/'individual'/('manual_'+n+'.json'),result)
    records.append((int(n),r,result))
for start in range(0,len(records),6):
    page=Image.new('RGB',(1440,1120),'white');draw=ImageDraw.Draw(page)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    for j,(n,r,res) in enumerate(records[start:start+6]):
        x=j%2*720;y=j//2*373
        with Image.open(root/r['image']) as im:
            original=im.convert('RGB').resize((336,336))
            fixed=overlay(im.convert('RGB'),res['objects']).resize((336,336))
        page.paste(original,(x+6,y+28));page.paste(fixed,(x+366,y+28))
        draw.text((x+6,y+4),'%03d %s | %d corrected ROIs'%(n,r['source_folder'],len(res['objects'])),font=font,fill='black')
    page.save(out/'sheets'/('manual_%02d.jpg'%(start//6+1)),quality=95)
print('Manual ROI images',len(records))
