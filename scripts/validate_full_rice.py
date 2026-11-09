# -*- coding: utf-8 -*-
"""Validate full automatic exports without interpreting predictions as truth."""
from pathlib import Path
import argparse,json,math,hashlib
from collections import Counter


def validate(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    ids={r['id']:r for r in manifest}
    assert len(ids)==len(manifest)==report['total_source_images_compared']
    label_count=0
    for r in manifest:
        assert (root/r['image']).is_file()
        assert (root/r['prediction_file']).is_file()
        if r['label'] is None:
            assert r['quality'] in {'unsupported_new_disease_no_semantic_mask','no_detection_not_a_negative_label'}
            assert r['default_split'] is None
            continue
        label_count+=1
        rows=(root/r['label']).read_text(encoding='utf-8').splitlines()
        if not rows:
            assert r['quality'] in {'ai_visually_screened_background','source_folder_background_unreviewed'}
        assert len(rows)==r['object_count']
        for line in rows:
            p=line.split();assert len(p)>=7 and (len(p)-1)%2==0
            cls=int(p[0]);assert 0<=cls<9
            nums=list(map(float,p[1:]));assert all(math.isfinite(x) and 0<=x<=1 for x in nums)
            xy=list(zip(nums[::2],nums[1::2]));assert len(set(xy))>=3
            area=abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1]-xy[i][1]*xy[(i+1)%len(xy)][0] for i in range(len(xy))))/2
            assert area>1e-9
        assert r['expert_diagnosis_confirmed'] is False
    paths={};groups={};hashes={}
    for name in ['train','val','all_prediction_train']:
        paths[name]=[(root/x).resolve() for x in (root/(name+'.txt')).read_text().splitlines() if x.strip()]
        assert len(paths[name])==len(set(paths[name]))
        groups[name]=set();hashes[name]=set()
        for p in paths[name]:
            r=ids[p.stem];assert r['label'] and p.is_file() and not r['exact_duplicate']
            groups[name].add(r['split_group']);hashes[name].add(r['sha256'])
            if name=='train':
                assert 'visual_review_previously_found_mask_problems' not in r['reasons']
                assert r['quality']!='source_folder_background_unreviewed'
            if name=='val':assert r['quality'].startswith('ai_visually_screened')
    assert not groups['train']&groups['val']
    assert not groups['all_prediction_train']&groups['val']
    assert not hashes['train']&hashes['val']
    assert label_count==report['source_files_with_model_or_checked_annotations']
    h=hashlib.sha256((root/'weights/best.pt').read_bytes()).hexdigest()
    assert h==report['primary_model_sha256']
    result={'structural_validation':'passed','source_images':len(manifest),
            'annotated_files':label_count,'train':len(paths['train']),'checked_val':len(paths['val']),
            'automatic_labels_are_ground_truth':False,'detected_split_group_leakage':False}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    validate(p.parse_args().root)
