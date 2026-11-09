"""Prompt SAM with visually selected symptom ROIs; outputs require inspection."""
from pathlib import Path
import json,time
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from ultralytics import SAM
from auto_label_rice import valid_polygon,write_json,digest_file
from repair_rice_annotations import overlay

root=Path('temp/rice_labeling/rice_archive_auto')
out=Path('temp/rice_labeling/archive_corrections')
rows=json.loads((out/'visual_selection.json').read_text())
rois=json.loads((out/'manual_rois.json').read_text())
points={1:[[75,108],[134,62],[148,99],[100,185]],
    7:[[73,44],[107,111],[147,171]],9:[[57,53],[143,120],[166,175]],
    21:[[104,7],[103,125],[204,206]],27:[[56,82],[100,80],[215,176]],32:[[174,106]]}
torch.set_num_threads(4)
weights=Path('temp/rice_labeling/models/sam2.1_b.pt')
model=SAM(str(weights));records=[]
for n,boxes in rois.items():
    r=rows[int(n)-1]
    expanded=[]
    for x0,y0,x1,y1 in boxes:
        dx=(x1-x0)*.22;dy=(y1-y0)*.22
        expanded.append([max(0,x0-dx),max(0,y0-dy),min(224,x1+dx),min(224,y1+dy)])
    result=model.predict(str(root/r['image']),bboxes=expanded,
        points=[[[x,y]] for x,y in points[int(n)]],labels=[[1]]*len(boxes),
        device='mps',verbose=False)[0]
    objects=[]
    if result.masks is not None:
        for polygon in result.masks.xyn:
            polygon=valid_polygon(polygon)
            if polygon:objects.append(dict(class_id=r['source_class_id'],polygon=polygon))
    record=dict(id=r['id'],sha256=r['sha256'],audit_index=int(n),objects=objects,
        method='visually_prompted_sam2.1_b_boundary_correction',
        reviewer='Codex visual inspection',expert_diagnosis_confirmed=False,
        visually_accepted=False,manual_boxes_224=boxes,manual_points_224=points[int(n)],
        classification_basis='source category and visible symptom morphology')
    write_json(out/'individual'/('sam_'+n+'.json'),record)
    records.append((int(n),r,record));print('SAM proposals',n,len(objects),flush=True)
for start in range(0,len(records),6):
    page=Image.new('RGB',(1440,1120),'white');draw=ImageDraw.Draw(page)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    for j,(n,r,res) in enumerate(records[start:start+6]):
        x=j%2*720;y=j//2*373
        with Image.open(root/r['image']) as im:
            original=im.convert('RGB').resize((336,336))
            fixed=overlay(im.convert('RGB'),res['objects']).resize((336,336))
        page.paste(original,(x+6,y+28));page.paste(fixed,(x+366,y+28))
        draw.text((x+6,y+4),'%03d %s | %d selected regions'%(n,r['source_folder'],len(res['objects'])),font=font,fill='black')
    page.save(out/'sheets'/('sam_%02d.jpg'%(start//6+1)),quality=95)
write_json(out/'sam_config.json',dict(model_sha256=digest_file(weights),
    model=str(weights),prompt_source='manual visual ROI selection',
    outputs_require_visual_review=True,no_original_checkpoint_modified=True))
