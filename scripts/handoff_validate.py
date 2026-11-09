# -*- coding: utf-8 -*-
"""Validate an extracted rice handoff without starting model training."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def rows(path, allow_background=False):
    objects = []
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.split()
        assert len(parts) >= 7 and (len(parts)-1) % 2 == 0, str(path)
        cls = int(parts[0])
        values = list(map(float, parts[1:]))
        assert 0 <= cls < 9 and all(math.isfinite(v) and 0 <= v <= 1 for v in values), str(path)
        xy = list(zip(values[::2], values[1::2]))
        area = abs(sum(xy[i][0]*xy[(i+1)%len(xy)][1] -
                       xy[(i+1)%len(xy)][0]*xy[i][1] for i in range(len(xy))))/2
        assert len(set(xy)) >= 3 and area > 1e-9, str(path)
        objects.append(cls)
    assert objects or allow_background, 'Empty positive annotation: '+str(path)
    return objects


def validate(root, load_images=False):
    root = Path(root).resolve()
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    by_id = {r['id']: r for r in manifest}
    assert len(by_id) == len(manifest) == report['source_images']
    assert len(list((root/'images/pool').glob('*.jpg'))) == len(manifest)
    ids, sha, groups = {}, {}, {}
    instances = Counter()
    for split in ['train', 'val']:
        image_paths = [(root/x).resolve() for x in (root/(split+'.txt')).read_text().splitlines() if x.strip()]
        assert image_paths and len(image_paths) == len(set(image_paths))
        ids[split], sha[split], groups[split] = set(), set(), set()
        for p in image_paths:
            assert p.is_file() and root in p.parents, str(p)
            r = by_id[p.stem]
            assert r['disposition'] == split and r['label'], p.stem
            background = r['annotation_quality'] == 'ai_visually_screened_background'
            assert (r['source_class_id'] < 9 or background) and r['reason'] is None
            classes = rows(root/r['label'], allow_background=background)
            assert (not classes and background) or all(c == r['source_class_id'] for c in classes), p.stem
            ids[split].add(p.stem)
            sha[split].add(r['source_sha256'])
            groups[split].add(r['split_group'])
            instances.update({split: len(classes)})
            if load_images:
                from PIL import Image
                with Image.open(p) as im:
                    im.load()
                    assert im.width > 1 and im.height > 1
        assert len(image_paths) == report[split+'_images' if split=='train' else 'validation_images']
    assert not ids['train'] & ids['val'], 'Shared IDs across splits'
    assert not sha['train'] & sha['val'], 'Exact duplicates across splits'
    assert not groups['train'] & groups['val'], 'Detected near-duplicate group across splits'
    for r in manifest:
        assert (root/r['image']).is_file()
        if r['disposition'] not in ['train','val']:
            assert r['id'] not in ids['train'] | ids['val']
            if r['disposition'] == 'duplicate':
                canonical = by_id[r['canonical_id']]
                assert r['source_sha256'] == canonical['source_sha256']
                assert r['source_class_id'] == canonical['source_class_id']
                assert (root/r['label']).read_bytes() == (root/canonical['label']).read_bytes()
            else:
                assert r['label'] is None
    h = hashlib.sha256()
    with (root/'weights/best.pt').open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    assert h.hexdigest() == report['model_sha256'], 'Model checkpoint hash mismatch'
    result = {'source_images_accounted_for': len(manifest), 'train': len(ids['train']),
              'val': len(ids['val']), 'polygon_instances': dict(instances),
              'detected_duplicate_leakage': False, 'structural_validation': 'passed',
              'label_quality': report['validation_reference']}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--load-images', action='store_true')
    args = parser.parse_args()
    validate(args.root, args.load_images)
