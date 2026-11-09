# -*- coding: utf-8 -*-
"""Compare every source image and export model-derived segmentation candidates.

This is a full inference/pseudo-label export, not a claim of complete ground
truth. It never manufactures a lesion for missing or unsupported predictions.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

import cv2
import numpy as np

from auto_label_rice import ORIGINAL_NAMES, digest_file, valid_polygon, write_json, yolo_text
from prepare_rice_training import copy_file, leakage_groups, read_polygons, yaml_text


def overlap(a, b, size=192):
    pa,pb=a['polygon'],b['polygon']
    if (max(x for x,y in pa)<min(x for x,y in pb) or
        max(x for x,y in pb)<min(x for x,y in pa) or
        max(y for x,y in pa)<min(y for x,y in pb) or
        max(y for x,y in pb)<min(y for x,y in pa)):
        return 0.,0.
    masks=[]
    for obj in [a,b]:
        mask=np.zeros((size,size),np.uint8)
        xy=np.rint(np.asarray(obj['polygon'])*(size-1)).astype(np.int32)
        cv2.fillPoly(mask,[xy],1)
        masks.append(mask)
    intersection=np.count_nonzero(masks[0]&masks[1])
    union=np.count_nonzero(masks[0]|masks[1])
    return intersection/max(1,union), intersection/max(1,min(m.sum() for m in masks))


def merge_predictions(named_predictions):
    """Keep actual predicted IDs; combine nonoverlapping masks, tag conflicts.

    Confidence scores from distinct models are not calibrated probabilities.
    Highest-score selection is a reproducible heuristic, not adjudication.
    """
    inputs=[]
    for model, objects in named_predictions:
        for obj in objects:
            if valid_polygon(obj['polygon']) is None:
                continue
            item=dict(obj)
            item['support']=[{'model':model,'score':obj['score'],'class_id':obj['class_id']}]
            inputs.append(item)
    selected=[]
    conflicts=0
    for obj in sorted(inputs,key=lambda o:o['score'],reverse=True):
        found=None
        for kept in selected:
            iou,contained=overlap(obj,kept)
            if iou>=.45 or (contained>=.85 and obj['class_id']==kept['class_id']):
                found=kept
                break
        if found is None:
            selected.append(obj)
        else:
            found['support'].extend(obj['support'])
            if found['class_id']!=obj['class_id']:
                conflicts+=1
    return selected, conflicts


def load_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def build(args):
    source=args.source.resolve()
    out=args.output.resolve()
    if out.exists():
        raise ValueError('Choose a new output directory; existing exports are preserved')
    records=[json.loads(x[0]) for x in sqlite3.connect(source/'progress.sqlite').execute(
        'select record from images order by folder,source')]
    if any(not r.get('image') for r in records):
        raise ValueError('An image has not been decoded/normalized')
    recovery_selection=load_json(args.recovery/'selection.json')
    if recovery_selection is None:
        raise ValueError('Missing recovery selection manifest')
    for sha in recovery_selection:
        for key in ['v8_best','v6_best']:
            if not (args.recovery/(sha+'_'+key+'.json')).exists():
                raise ValueError('High-resolution recovery is still incomplete: '+sha)
    by_sha=defaultdict(list)
    for r in records:
        by_sha[r['sha256']].append(r)
    groups=leakage_groups(records,source,source.parent/'phash_cache.json')
    checked_manifest=load_json(args.checked/'manifest.json')
    checked={r['source_sha256']:r for r in checked_manifest}
    manual={r['source_sha256']:r for r in load_json(args.manual,[])
            if r.get('visually_accepted') is True}
    backgrounds={r['source_sha256']:r for r in load_json(args.backgrounds)}
    review=load_json(args.decisions)
    bad_ids={r['id'] for r in review if r['status'].startswith('needs_')}
    bad_groups={groups[r['sha256']] for r in records if r['id'] in bad_ids}
    old_manifest=load_json(args.curated/'manifest.json')
    val_groups={r['split_group'] for r in old_manifest if r['disposition']=='val'}
    val_ids={r['id'] for r in old_manifest if r['disposition']=='val'}
    for name in ['images/pool','labels/pool','predictions','weights']:
        (out/name).mkdir(parents=True,exist_ok=True)
    canonical_annotations={}
    for sequence,(sha,rs) in enumerate(by_sha.items(),1):
        r=rs[0]
        known=r['folder_class_id'] is not None and r['folder_class_id']<9
        primary=load_json(args.additional/(sha+'_v8.json'),r['objects'])
        secondary=load_json(args.additional/(sha+'_v6.json'),
                            load_json(args.ensemble/(sha+'.json'),[]))
        hi_a=load_json(args.recovery/(sha+'_v8_best.json'),[])
        hi_b=load_json(args.recovery/(sha+'_v6_best.json'),[])
        if known and not (args.ensemble/(sha+'.json')).exists():
            raise ValueError('Missing second-model comparison for '+sha)
        if not known and (not (args.additional/(sha+'_v8.json')).exists() or
                          not (args.additional/(sha+'_v6.json')).exists()):
            raise ValueError('Comparison of new/healthy source is incomplete: '+sha)
        named=[('v8_800',primary),('v6_800',secondary)]
        # Standard-resolution predictions take priority over weak recovery.
        # Recovery fills misses; existing failed masks are shown for repair,
        # and remain excluded from the default train list.
        if not (primary or secondary):
            named.extend([('v8_1280_recovery',hi_a),('v6_1280_recovery',hi_b)])
        objects,conflicts=merge_predictions(named)
        quality='automatic_pseudo_label_unreviewed'
        reasons=[]
        if len({x['folder_class_id'] for x in rs})>1:
            reasons.append('source_folders_disagree_on_class')
        if not known and r['folder_class_id']!=12:
            objects=[]
            quality='unsupported_new_disease_no_semantic_mask'
            reasons.append('nine_class_models_cannot_label_this_new_disease')
        elif r['folder_class_id']==12:
            objects=[]
            screened=backgrounds.get(sha,{}).get('accepted_background',False)
            quality='ai_visually_screened_background' if screened else 'source_folder_background_unreviewed'
            if not screened:
                reasons.append('healthy_source_image_not_visually_accepted')
        elif sha in manual:
            objects=manual[sha]['objects']
            if any(valid_polygon(o['polygon']) is None for o in objects):
                raise ValueError('Invalid manually reviewed mask')
            quality='ai_visually_screened_mask'
        elif sha in checked:
            meta=checked[sha]
            objects=read_polygons(args.checked/'labels'/(meta['id']+'.txt'))
            quality='ai_visually_screened_mask'
        elif not objects:
            quality='no_detection_not_a_negative_label'
            reasons.append('requires_actual_annotation_no_mask_available')
        else:
            if groups[sha] in bad_groups:
                reasons.append('visual_review_previously_found_mask_problems')
            if any(o['class_id']!=r['folder_class_id'] for o in objects):
                reasons.append('predicted_class_differs_from_source_folder')
            if min(o['score'] for o in objects)<.25:
                reasons.append('low_confidence_high_resolution_recovery')
            if conflicts:
                reasons.append('models_disagree_on_overlapping_object_class')
        canonical_annotations[sha]={'objects':objects,'quality':quality,'reasons':reasons,
                                    'raw':{'v8_800':primary,'v6_800':secondary,
                                           'v8_1280':hi_a,'v6_1280':hi_b},
                                    'overlapping_class_conflicts':conflicts}
        if sequence%250==0:
            print(json.dumps({'compared_unique_images':sequence,'total':len(by_sha)}),flush=True)
    manifest=[]
    train=[]; val=[]; raw_train=[]
    counts=Counter(); unique_counts=Counter()
    train_class_counts=Counter()
    model_counts={'v8_800':Counter(),'v6_800':Counter()}
    for sha,rs in by_sha.items():
        a=canonical_annotations[sha]
        known_label=bool(a['objects']) or a['quality'] in {
            'ai_visually_screened_background','source_folder_background_unreviewed'}
        blocked='visual_review_previously_found_mask_problems' in a['reasons'] or a['quality']=='source_folder_background_unreviewed'
        for i,r in enumerate(rs):
            name=r['id']
            copy_file(source/r['image'],out/'images/pool'/(name+'.jpg'))
            label='labels/pool/'+name+'.txt' if known_label else None
            if label:
                (out/label).write_text(yolo_text(a['objects']),encoding='utf-8')
            split=None
            if i==0 and label:
                path='./images/pool/'+name+'.jpg'
                if groups[sha] in val_groups:
                    # Use only previously screened val references, not pseudo
                    # labels or additional near duplicates, for validation.
                    if name in val_ids:
                        val.append(path);split='val'
                else:
                    raw_train.append(path)
                    if not blocked:
                        train.append(path);split='train'
                        train_class_counts.update(o['class_id'] for o in a['objects'])
            counts[a['quality']]+=1
            if i==0:unique_counts[a['quality']]+=1
            record={
                'id':name,'source':r['source'],'source_folder':r['folder'],
                'source_class_id':r['folder_class_id'],'sha256':sha,
                'split_group':groups[sha],'canonical_id':rs[0]['id'],
                'image':'images/pool/'+name+'.jpg','label':label,
                'quality':a['quality'],'reasons':a['reasons'],
                'default_split':split,'exact_duplicate':i>0,
                'prediction_file':'predictions/'+name+'.json',
                'object_count':len(a['objects']),
                'class_ids':sorted({o['class_id'] for o in a['objects']}),
                'expert_diagnosis_confirmed':False,
            }
            manifest.append(record)
            for model,stats in model_counts.items():
                predictions=a['raw'][model]
                stats['source_files_compared']+=1
                stats['source_files_with_detections']+=bool(predictions)
                stats['polygon_instances']+=len(predictions)
                if r['folder_class_id'] is not None and r['folder_class_id']<9:
                    stats['known_disease_source_files']+=1
                    if predictions:
                        top=max(predictions,key=lambda x:x['score'])
                        stats['top_predicted_class_matches_source_folder']+=top['class_id']==r['folder_class_id']
            write_json(out/record['prediction_file'],{
                'id':name,'source_folder_class_id':r['folder_class_id'],
                'objects':a['objects'],'quality':a['quality'],'reasons':a['reasons'],
                'raw_models':a['raw'],'expert_diagnosis_confirmed':False,
                'visual_review_provenance':manual.get(sha,checked.get(sha,backgrounds.get(sha))),
            })
    if not train or not val:
        raise ValueError('No usable train or checked val list')
    for filename,paths in [('train.txt',train),('val.txt',val),('all_prediction_train.txt',raw_train)]:
        (out/filename).write_text('\n'.join(paths)+'\n',encoding='utf-8')
    (out/'data.yaml').write_text(yaml_text('train.txt','val.txt'),encoding='utf-8')
    (out/'data_all_predictions.yaml').write_text(yaml_text('all_prediction_train.txt','val.txt'),encoding='utf-8')
    write_json(out/'manifest.json',manifest)
    import csv
    with (out/'image_status.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','source','source_folder','source_class_id','quality','object_count','label','default_split','reasons']
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in manifest:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],list) else r[k] for k in fields})
    result={
        'total_source_images_compared':len(records),'unique_file_contents':len(by_sha),
        'source_files_with_model_or_checked_annotations':sum(r['label'] is not None for r in manifest),
        'source_files_without_semantic_annotations':sum(r['label'] is None for r in manifest),
        'source_file_quality_counts':dict(counts),'unique_content_quality_counts':dict(unique_counts),
        'default_train_images':len(train),'checked_validation_images':len(val),
        'all_prediction_train_images_including_known_problem_masks':len(raw_train),
        'default_train_polygon_instances_by_class':dict(train_class_counts),
        'class_names':ORIGINAL_NAMES,
        'standard_model_comparison_counts_not_accuracy':{k:dict(v) for k,v in model_counts.items()},
        'recovery_settings':load_json(args.recovery/'config.json'),
        'primary_model_sha256':digest_file(args.model),
        'secondary_model_sha256':load_json(args.ensemble/'config.json')['model_sha256'],
        'all_source_images_have_training_ready_ground_truth':False,
        'evaluation_limit':'Validation is AI-screened and covers only three positive disease classes; merge original val/test for meaningful nine-class evaluation.',
        'label_policy':'Preserve actual teacher-predicted IDs; do not force a source-folder class or label missing detections as healthy.',
        'background_policy':'Empty labels only for source Healthy images; default train requires previous AI visual acceptance.',
    }
    write_json(out/'report.json',result)
    copy_file(args.model,out/'weights/best.pt')
    shutil.copyfile(Path(__file__).parent/'merge_rice_training.py',out/'merge_rice_training.py')
    shutil.copyfile(Path(__file__).parent/'validate_full_rice.py',out/'validate_full_rice.py')
    shutil.copyfile(Path(__file__).parent/'rice_model_compare.html',out/'compare.html')
    shutil.copyfile(Path(__file__).parent/'RICE_FULL_AUTO_README.md',out/'README_TH.md')
    (out/'requirements.txt').write_text('ultralytics==8.4.137\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


def package(output):
    output=output.resolve()
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',allowZip64=True) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix not in {'.cache','.pyc'} and '__pycache__' not in path.parts:
                archive.write(path,Path(output.name)/path.relative_to(output),
                              compress_type=zipfile.ZIP_STORED if path.suffix in {'.jpg','.pt'} else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
        assert archive.testzip() is None
    print(json.dumps({'zip':str(output.with_suffix('.zip')),'bytes':output.with_suffix('.zip').stat().st_size}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('temp/rice_labeling/local_all'))
    p.add_argument('--ensemble',type=Path,default=Path('temp/rice_labeling/ensemble'))
    p.add_argument('--recovery',type=Path,default=Path('temp/rice_labeling/recovery'))
    p.add_argument('--additional',type=Path,default=Path('temp/rice_labeling/additional_baseline'))
    p.add_argument('--checked',type=Path,default=Path('temp/rice_labeling/visually_checked_yolo'))
    p.add_argument('--manual',type=Path,default=Path('temp/rice_labeling/manual_recovery/accepted.json'))
    p.add_argument('--curated',type=Path,default=Path('temp/rice_labeling/rice_training_handoff'))
    p.add_argument('--backgrounds',type=Path,default=Path('temp/rice_labeling/new_class_audit/background_review.json'))
    p.add_argument('--decisions',type=Path,default=Path('temp/rice_labeling/visual_audit/review_decisions.json'))
    p.add_argument('--model',type=Path,default=Path('best.pt'))
    p.add_argument('--output',type=Path,default=Path('temp/rice_labeling/rice_full_auto'))
    p.add_argument('--package-only',action='store_true')
    args=p.parse_args()
    if not args.package_only:build(args)
    package(args.output)
