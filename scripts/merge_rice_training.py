# -*- coding: utf-8 -*-
"""Combine extra training images with an available original nine-class dataset.

Keeps the original validation/test references. Does not train, resume a past
optimizer run, download data, or change any model/checkpoint/source annotation.
"""
import argparse
import json
from pathlib import Path


NAMES = ['Bacterial_Blight', 'Brown_Spot', 'Rice_Blast', 'Narrow_Brown_Spot',
         'False_Smut', 'Dirty_Seed', 'Sheath_Rot', 'Stem_Rot', 'Red_Stripe']


def image_files(value, root):
    """Resolve standard YOLO image directories/lists, including ./ in lists."""
    extensions = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp','.avif','.heic'}
    result = set()
    for part in value if isinstance(value, list) else [value]:
        p = Path(part)
        if not p.is_absolute():
            p = root/p
        p = p.resolve()
        if p.is_dir():
            result.update(x.resolve() for x in p.rglob('*') if x.suffix.lower() in extensions)
        elif p.suffix.lower() == '.txt' and p.is_file():
            for line in p.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                q = Path(line.strip())
                result.add(q.resolve() if q.is_absolute() else (p.parent/q).resolve())
        elif p.is_file() and p.suffix.lower() in extensions:
            result.add(p)
        else:
            raise FileNotFoundError('Missing image directory/list: '+str(p))
    for p in result:
        if not p.is_file():
            raise FileNotFoundError('Missing image: '+str(p))
    if not result:
        raise ValueError('No images found')
    return result


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-yaml', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    bundle = args.root.resolve()
    source_yaml = args.original_yaml.resolve()
    original = yaml.safe_load(source_yaml.read_text(encoding='utf-8'))
    names = original.get('names')
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != NAMES:
        raise ValueError('Original class IDs/names must exactly match the nine-class checkpoint')
    data_root = Path(original.get('path') or source_yaml.parent)
    if not data_root.is_absolute():
        data_root = (source_yaml.parent/data_root).resolve()
    old_train = image_files(original['train'], data_root)
    old_val = image_files(original['val'], data_root)
    old_test = image_files(original['test'], data_root) if original.get('test') else set()
    if old_train & (old_val | old_test):
        raise ValueError('Original dataset reuses paths between train and evaluation')
    additions = image_files('train.txt', bundle)
    # Compare perceptual hashes against old val/test to keep detected variants
    # out of the merged training set. No field/collection identity is inferred.
    import cv2
    import numpy as np
    from PIL import Image
    def hashes(path):
        with Image.open(path) as im:
            gray = cv2.cvtColor(np.asarray(im.convert('RGB').resize((32,32))), cv2.COLOR_RGB2GRAY)
        result = set()
        for flip in [gray, np.fliplr(gray)]:
            for k in range(4):
                low = cv2.dct(np.float32(np.rot90(flip,k)))[:8,:8].ravel()[1:]
                result.add(int(''.join('1' if x else '0' for x in low > np.median(low)),2))
        return result
    eval_hashes = {h for p in old_val | old_test for h in hashes(p)}
    held, accepted = [], []
    for p in sorted(additions):
        near = any(bin(a^b).count('1') <= 4 for a in hashes(p) for b in eval_hashes)
        (held if near else accepted).append(p)
    if not accepted:
        raise ValueError('No additional training images remain after evaluation duplicate screening')
    out_dir = bundle/'merged_with_original'
    out_dir.mkdir(exist_ok=True)
    merged_train = sorted(old_train | set(accepted))
    for name, paths in [('train.txt',merged_train),('val.txt',sorted(old_val)),('test.txt',sorted(old_test))]:
        if paths:
            (out_dir/name).write_text('\n'.join(map(str,paths))+'\n',encoding='utf-8')
    result = {'train':'train.txt', 'val':'val.txt', 'nc':9, 'names':dict(enumerate(NAMES))}
    if old_test:
        result['test']='test.txt'
    (out_dir/'data.yaml').write_text(yaml.safe_dump(result,sort_keys=False),encoding='utf-8')
    report = {'original_train':len(old_train), 'added_train':len(accepted),
              'additions_held_as_evaluation_near_duplicates':list(map(str,held)),
              'merged_train':len(merged_train), 'original_val_unchanged':len(old_val),
              'original_test_unchanged':len(old_test),
              'limit':'perceptual screening reduces detected leakage; no collection metadata available'}
    (out_dir/'merge_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'data_yaml':str(out_dir/'data.yaml'),**report},indent=2))


if __name__=='__main__':
    main()
