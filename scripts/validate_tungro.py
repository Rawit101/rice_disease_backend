"""Check class mapping, annotations, split isolation, and optional YOLO loading.

Structural validation does not confirm a disease diagnosis or mask accuracy.
"""
from pathlib import Path
import argparse,hashlib,json,math

EXPECTED=['Bacterial_Blight','Brown_Spot','Rice_Blast','Narrow_Brown_Spot','False_Smut',
          'Dirty_Seed','Sheath_Rot','Stem_Rot','Red_Stripe','Tungro']


def label_rows(path):
    result=[]
    for line in path.read_text().splitlines():
        p=line.split();assert len(p)>=7 and (len(p)-1)%2==0
        cls=int(p[0]);assert 0<=cls<10
        a=list(map(float,p[1:]));assert all(math.isfinite(x) and 0<=x<=1 for x in a)
        xy=list(zip(a[::2],a[1::2]));assert len(set(xy))>=3
        area=abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1]-xy[i][1]*xy[(i+1)%len(xy)][0] for i in range(len(xy))))/2
        assert area>1e-9;result.append(cls)
    return result


def validate(root,load=False):
    root=Path(root).resolve()
    read=lambda n:json.loads((root/n).read_text())
    report=read('report.json');m=read('manifest.json');base=read('base_manifest.json')
    assert report['class_names']==EXPECTED and read('class_mapping.json')['names']==EXPECTED
    assert len(m)==report['total_tungro_images']==3228
    byid={r['id']:r for r in base+m};assert len(byid)==len(base)+len(m)
    label_count=0
    for r in m:
        assert r['export_class_id']==r['source_class_id']==9
        assert r['source_taxonomy_class_id']==19
        assert (root/r['image']).is_file() and (root/r['prediction_file']).is_file()
        d=json.loads((root/r['prediction_file']).read_text())
        assert r['expert_diagnosis_confirmed'] is False
        assert d['sam_object_support_executed'] is True,'Incomplete object-support pass'
        if r['label']:
            c=label_rows(root/r['label']);assert c and set(c)=={9}
            assert len(c)==r['object_count']==len(d['objects']);label_count+=1
        else:
            assert not d['objects'] and r['object_count']==0
            assert r['default_split'] is None and r['draft_split'] is None
    assert label_count==report['tungro_label_files']
    pools={};groups={};sha={}
    for split in ['train','train_draft','val']:
        paths=[x for x in (root/(split+'.txt')).read_text().splitlines() if x.strip()]
        assert len(paths)==len(set(paths));pools[split]=set(paths);groups[split]=set();sha[split]=set()
        for path in paths:
            p=(root/path).resolve();assert p.is_file() and root in p.parents
            r=byid[p.stem];assert r['label'];classes=label_rows(root/r['label'])
            if r.get('export_class_id')==9:
                assert split!='val' and not r['reasons'] and classes and set(classes)=={9}
                if split=='train':assert r['quality']=='ai_visually_reviewed_symptom_mask'
            else:assert all(c<9 for c in classes)
            groups[split].add(r['split_group']);sha[split].add(r['sha256'])
    assert pools['train']<=pools['train_draft']
    for split in ['train','train_draft']:
        assert not pools[split]&pools['val'];assert not groups[split]&groups['val'];assert not sha[split]&sha['val']
    assert len(pools['train'])==report['default_train_images']
    assert len(pools['train_draft'])==report['draft_train_images']
    assert len(pools['val'])==report['validation_images']
    assert hashlib.sha256((root/'weights/best.pt').read_bytes()).hexdigest()==report['primary_model_sha256']
    result=dict(structural_validation='passed',tungro_images=len(m),tungro_label_files=label_count,
        default_train_images=len(pools['train']),draft_train_images=len(pools['train_draft']),
        validation_images=len(pools['val']),tungro_validation_images=0,
        old_class_ids_preserved=True,new_tungro_class_id=9,disease_ground_truth_verified=False,training_started=False)
    if load:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset
        data=check_det_dataset(str(root/'data.yaml'),autodownload=False)
        result['loader']={}
        for split in ['train','val']:
            ds=YOLODataset(img_path=data[split],imgsz=800,batch_size=4,data=data,task='segment',augment=False,rect=False,cache=False)
            assert len(ds)==len(pools[split])
            assert sum(len(x['cls']) for x in ds.labels)==sum(len(x['segments']) for x in ds.labels)
            batch=ds.collate_fn([ds[i] for i in range(min(4,len(ds)))])
            assert tuple(batch['img'].shape[1:])==(3,800,800)
            result['loader'][split]=dict(images=len(ds),segments=sum(len(x['segments']) for x in ds.labels),batch_shape=list(batch['img'].shape))
    (root/'validation_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--load-yolo',action='store_true')
    a=p.parse_args();validate(a.root,a.load_yolo)
