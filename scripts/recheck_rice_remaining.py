# -*- coding: utf-8 -*-
"""Recheck the fixed unresolved subset; keep crop proposals out of training."""
from pathlib import Path
from collections import Counter
import argparse,json,time

import numpy as np
from PIL import Image

from auto_label_rice import check_model_names, digest_file, polygons_from_result, valid_polygon, write_json
from build_full_rice_comparison import merge_predictions


def crop_boxes(width,height):
    cw,ch=max(1,round(width*.65)),max(1,round(height*.65))
    return list(dict.fromkeys([(0,0,cw,ch),(width-cw,0,width,ch),
        (0,height-ch,cw,height),(width-cw,height-ch,width,height),
        ((width-cw)//2,(height-ch)//2,(width+cw)//2,(height+ch)//2)]))


def map_crop_object(obj,box,width,height):
    x0,y0,x1,y1=box
    polygon=valid_polygon([[(x0+x*(x1-x0))/width,(y0+y*(y1-y0))/height]
                           for x,y in obj['polygon']])
    if polygon is None:return None
    local=np.asarray(obj['polygon'])
    truncated=bool((x0>0 and local[:,0].min()<.012) or
        (y0>0 and local[:,1].min()<.012) or
        (x1<width and local[:,0].max()>.988) or
        (y1<height and local[:,1].max()>.988))
    return {**obj,'polygon':polygon,'crop_box':list(box),'touches_internal_crop_edge':truncated}


def inspect(args):
    rows=[r for r in json.loads((args.source/'manifest.json').read_text()) if r['label'] is None]
    args.output.mkdir(parents=True,exist_ok=True)
    saved=args.output/'selection.json'
    if saved.exists():
        old=json.loads(saved.read_text())
        if [r['id'] for r in old]!=[r['id'] for r in rows]:
            raise ValueError('Pending subset changed; use a new output directory')
    else:write_json(saved,rows)
    quality={}
    for r in rows:
        if r['sha256'] in quality:continue
        with Image.open(args.source/r['image']) as im:
            im.load();rgb=np.asarray(im.convert('RGB'))
            lit=(rgb.max(axis=2)>30)
            quality[r['sha256']]={'width':im.width,'height':im.height,
                'decoded':True,'pixels_above_30':int(lit.sum()),
                'nearly_empty_black_crop':int(lit.sum())<8,
                'small_source_resolution':min(im.size)<=224}
        with Image.open(r['source']) as original:
            original.load()
            quality[r['sha256']]['original_dimensions']=list(original.size)
            quality[r['sha256']]['original_format']=original.format
    write_json(args.output/'quality.json',quality)
    print(json.dumps({'pending_files':len(rows),'unique_contents':len(quality),
        'small_sources':sum(x['small_source_resolution'] for x in quality.values()),
        'nearly_empty_black_contents':sum(x['nearly_empty_black_crop'] for x in quality.values())}),flush=True)
    return rows,quality


def infer(args,rows,quality):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    models=[('v8',args.primary),('v6',args.secondary)]
    config={'models':{k:digest_file(p) for k,p in models},'imgsz':800,'conf':.25,
            'crop_fraction':.65,'reviewed':False,'proposal_only_not_training_labels':True}
    existing=args.output/'config.json'
    if existing.exists() and json.loads(existing.read_text())!=config:
        raise ValueError('Recheck parameters changed; use a new output directory')
    write_json(existing,config)
    unique={r['sha256']:r for r in rows if r['source_class_id']<9 and
            not quality[r['sha256']]['nearly_empty_black_crop']}
    cache=args.output/'cache';cache.mkdir(exist_ok=True)
    started=time.monotonic()
    for name,path in models:
        model=YOLO(str(path));check_model_names(model)
        model.predict(np.zeros((800,800,3),np.uint8),imgsz=800,device=device,verbose=False)
        pending=[r for sha,r in unique.items() if not (cache/(sha+'_'+name+'.json')).exists()]
        for i,r in enumerate(pending):
            im=Image.open(args.source/r['image']).convert('RGB')
            predictions=[]
            for box in crop_boxes(im.width,im.height):
                result=model.predict(im.crop(box),imgsz=800,conf=.25,device=device,
                                     max_det=100,verbose=False)[0]
                for obj in polygons_from_result(result,True):
                    mapped=map_crop_object(obj,box,im.width,im.height)
                    if mapped is not None:predictions.append(mapped)
            write_json(cache/(r['sha256']+'_'+name+'.json'),predictions)
            if (i+1)%10==0 or i+1==len(pending):
                print(json.dumps({'model':name,'images':i+1,'total':len(pending),
                                  'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
        del model
        if device=='mps':torch.mps.empty_cache()


def report(args,rows,quality):
    merged={};counts=Counter()
    for r in rows:
        sha=r['sha256']
        if sha in merged:continue
        candidates=[]
        if r['source_class_id']<9:
            for model in ['v8','v6']:
                p=args.output/'cache'/(sha+'_'+model+'.json')
                if not p.exists() and not quality[sha]['nearly_empty_black_crop']:
                    raise ValueError('Crop inference is incomplete: '+sha)
                objects=json.loads(p.read_text()) if p.exists() else []
                candidates.append((model,objects))
        objects,conflicts=merge_predictions(candidates)
        status=('unsupported_new_class' if r['source_class_id']>=9 else
                'nearly_empty_black_crop' if quality[sha]['nearly_empty_black_crop'] else
                'crop_candidates_need_review' if objects else 'no_usable_crop_prediction')
        merged[sha]={'status':status,'quality':quality[sha],'objects':objects,
                     'overlap_class_conflicts':conflicts,'is_training_label':False}
    for r in rows:counts[merged[r['sha256']]['status']]+=1
    write_json(args.output/'results.json',merged)
    write_json(args.output/'summary.json',{'checked_files':len(rows),'unique_contents':len(merged),
        'file_counts':dict(counts),'new_training_labels_automatically_approved':0})
    print(json.dumps(dict(counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('temp/rice_labeling/rice_full_auto'))
    p.add_argument('--output',type=Path,default=Path('temp/rice_labeling/remaining_recheck'))
    p.add_argument('--primary',type=Path,default=Path('best.pt'))
    p.add_argument('--secondary',type=Path,default=Path('/Users/ssrawit/Downloads/rice9disease_v6_BG_Polygon/weights/best.pt'))
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();rows,quality=inspect(a)
    if not a.report_only:infer(a,rows,quality)
    report(a,rows,quality)
