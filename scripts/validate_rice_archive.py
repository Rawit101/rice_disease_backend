"""Structural validation only; this cannot certify disease or mask accuracy."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label_rows(path):
    classes=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        fields=line.split()
        assert len(fields)>=7 and (len(fields)-1)%2==0, str(path)
        cls=int(fields[0]); assert 0<=cls<9
        v=list(map(float,fields[1:]))
        assert all(math.isfinite(x) and 0<=x<=1 for x in v)
        xy=list(zip(v[::2],v[1::2]))
        assert len(set(xy))>=3
        area=abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1]-xy[i][1]*xy[(i+1)%len(xy)][0] for i in range(len(xy))))/2
        assert area>1e-9
        classes.append(cls)
    return classes


def validate(root):
    root=Path(root).resolve()
    report=read(root/'report.json'); manifest=read(root/'manifest.json')
    previous_val=read(root/'validation_manifest.json')
    by_id={r['id']:r for r in manifest}
    val_ids={r['id']:r for r in previous_val}
    assert len(manifest)==len(by_id)==report['total_source_images_compared']
    assert len(val_ids)==report['prior_checked_validation_images']
    labels=0
    for r in manifest:
        assert r['expert_diagnosis_confirmed'] is False
        if r['quality']=='read_error':
            assert not r['label'] and r['default_split'] is None
            continue
        assert (root/r['image']).is_file() and (root/r['thumbnail']).is_file()
        assert (root/r['prediction_file']).is_file()
        if r['source_class_id'] is None or r['source_class_id']>=9:
            assert (r['label'] is None and r['default_split'] is None) or (
                r['source_class_id']==12 and r['quality']=='ai_visually_screened_background')
        if r['label']:
            classes=label_rows(root/r['label'])
            assert len(classes)==r['object_count']
            assert classes or r['quality']=='ai_visually_screened_background'
            labels+=1
        else:
            assert r['object_count']==0 and r['default_split'] is None
    assert labels==report['source_files_with_model_or_checked_annotations']
    assert len(manifest)-labels==report['source_files_without_semantic_annotations']
    paths={}; hashes={}; groups={}; class_counts={}
    for name in ['train','strict_train','all_prediction_train','val']:
        paths[name]=[(root/x).resolve() for x in (root/(name+'.txt')).read_text().splitlines() if x.strip()]
        assert len(paths[name])==len(set(paths[name]))
        hashes[name]=set();groups[name]=set();classes=Counter()
        for p in paths[name]:
            assert p.is_file() and root in p.parents
            r=(val_ids if name=='val' else by_id)[p.stem]
            assert r['label']
            c=label_rows(root/r['label'])
            classes.update(c)
            if name=='val':
                assert r['quality'].startswith('ai_visually_screened')
            else:
                assert not r['exact_duplicate']
                assert r['sha256'] not in hashes[name]
                if name in ['train','strict_train']:
                    assert r['default_split']=='train' and not r['reasons']
                    assert (c and all(x==r['source_class_id'] for x in c)) or (
                        not c and r['quality']=='ai_visually_screened_background')
                if name=='strict_train':
                    assert r['two_model_agreement_candidate'] is True
            hashes[name].add(r['sha256']);groups[name].add(r['split_group'])
        class_counts[name]=dict(classes)
    assert set(paths['strict_train'])<=set(paths['train'])<=set(paths['all_prediction_train'])
    for name in ['train','strict_train','all_prediction_train']:
        assert not groups[name]&groups['val'], 'Near duplicate leakage: '+name
        assert not hashes[name]&hashes['val'], 'Exact duplicate leakage: '+name
    assert len(paths['train'])==report['default_train_images']
    assert len(paths['strict_train'])==report['two_model_agreement_train_images']
    assert len(paths['all_prediction_train'])==report['raw_prediction_train_images']
    sha=hashlib.sha256((root/'weights/best.pt').read_bytes()).hexdigest()
    assert sha==report['primary_model_sha256']
    result=dict(structural_validation='passed',source_images=len(manifest),
        annotated_files=labels, train_images=len(paths['train']),
        strict_train_images=len(paths['strict_train']), checked_val_images=len(paths['val']),
        polygon_classes=class_counts,detected_split_leakage=False,
        labels_are_expert_ground_truth=False)
    print(json.dumps(result,indent=2),flush=True)
    (root/'validation_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def validate_loader(root):
    """Load the default train/val and one CPU batch; never start training."""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    root=Path(root).resolve()
    report=read(root/'report.json')
    data=check_det_dataset(str(root/'data.yaml'),autodownload=False)
    result={}
    for split in ['train','val']:
        dataset=YOLODataset(img_path=data[split],imgsz=800,batch_size=4,
            data=data,task='segment',augment=False,rect=False,cache=False)
        assert len(dataset)>0
        assert len(dataset)==report['default_train_images' if split=='train' else 'prior_checked_validation_images']
        instances=sum(len(x['cls']) for x in dataset.labels)
        segments=sum(len(x['segments']) for x in dataset.labels)
        assert instances==segments
        batch=dataset.collate_fn([dataset[i] for i in range(min(4,len(dataset)))])
        assert tuple(batch['img'].shape[1:])==(3,800,800)
        result[split]=dict(images=len(dataset),segments=segments,
            batch_shape=list(batch['img'].shape),task='segment')
    result['training_started']=False
    (root/'loader_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(yolo_loader_validation='passed',**result),indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--load-yolo',action='store_true')
    args=p.parse_args();validate(args.root)
    if args.load_yolo:validate_loader(args.root)
