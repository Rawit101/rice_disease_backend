# -*- coding: utf-8 -*-
"""Package reviewed masks and conservative two-model pseudo-labels for YOLO.

No inference result is promoted to human/expert ground truth. Every input is
accounted for, but unsupported, ambiguous and failed masks stay out of training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import zipfile

import cv2
import numpy as np
from PIL import Image

from auto_label_rice import ORIGINAL_NAMES, digest_file, valid_polygon, write_json, yolo_text


class Groups:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def mask_for(obj, size=256):
    mask = np.zeros((size, size), np.uint8)
    points = np.rint(np.array(obj['polygon']) * (size-1)).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def agreement(first, second, expected, min_score=.5, min_iou=.5):
    """Class/count/instance agreement is a selection rule, not truth assurance."""
    if not first or not second:
        return False, 'one_or_both_models_missing_labels'
    if any(o['class_id'] != expected for o in first+second):
        return False, 'model_class_disagrees_with_source'
    if any(o['score'] < min_score for o in first+second):
        return False, 'model_score_below_selection_threshold'
    if len(first) != len(second):
        return False, 'models_disagree_on_instance_count'
    masks_a, masks_b = [mask_for(o) for o in first], [mask_for(o) for o in second]
    pairs = []
    for i, a in enumerate(masks_a):
        for j, b in enumerate(masks_b):
            union = np.count_nonzero(a | b)
            iou = np.count_nonzero(a & b) / max(1, union)
            pairs.append((iou, i, j))
    used_a, used_b = set(), set()
    for iou, i, j in sorted(pairs, reverse=True):
        if iou >= min_iou and i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return (True, 'two_model_pseudo_label') if len(used_a) == len(first) else (
        False, 'models_disagree_on_mask_boundary')


def phash_variants(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB').resize((32, 32)), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hashes = set()
    for flip in [gray, np.fliplr(gray)]:
        for rotation in range(4):
            low = cv2.dct(np.float32(np.rot90(flip, rotation)))[:8, :8].ravel()
            bits = low[1:] > np.median(low[1:])
            hashes.add(int(''.join('1' if b else '0' for b in bits), 2))
    return sorted(hashes)


def leakage_groups(records, root, cache_path):
    """Group exact duplicates and conservative flip/rotation pHash matches.

    This reduces leakage among detected variants; it cannot establish field,
    photographer or collection independence without missing source metadata.
    """
    canonical = {r['sha256']: r for r in records}
    groups = Groups(canonical)
    saved = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    bands = defaultdict(list)
    new = 0
    for sha, r in canonical.items():
        if sha not in saved:
            saved[sha] = phash_variants(root/r['image'])
            new += 1
        for value in saved[sha]:
            candidates = set()
            # Five disjoint bands guarantee one equal band for <=4 bit changes.
            for band in range(5):
                key = (band, (value >> (band*13)) & 8191)
                candidates.update(bands[key])
            for other, other_value in candidates:
                if other != sha and bin(value ^ other_value).count('1') <= 4:
                    groups.join(sha, other)
            for band in range(5):
                bands[(band, (value >> (band*13)) & 8191)].append((sha, value))
    if new:
        write_json(cache_path, saved)
    return {sha: groups.find(sha) for sha in canonical}


def read_polygons(path):
    objects = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 7 or (len(parts)-1) % 2:
            raise ValueError('Malformed segmentation row: '+str(path))
        class_id = int(parts[0])
        xy = list(map(float, parts[1:]))
        polygon = valid_polygon(list(zip(xy[::2], xy[1::2])))
        if polygon is None or not 0 <= class_id < len(ORIGINAL_NAMES):
            raise ValueError('Invalid segmentation polygon: '+str(path))
        objects.append({'class_id': class_id, 'polygon': polygon})
    if not objects:
        raise ValueError('Empty positive annotation: '+str(path))
    return objects


def copy_file(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)


def yaml_text(train, val):
    # Omit path: so Ultralytics resolves the dataset against this YAML's parent.
    return ('train: '+train+'\nval: '+val+'\nnc: 9\nnames:\n'+
            ''.join(f'  {i}: {name}\n' for i, name in enumerate(ORIGINAL_NAMES)))


def build(args):
    root, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Output exists; choose a new directory to preserve prior exports')
    records = [json.loads(x[0]) for x in sqlite3.connect(root/'progress.sqlite').execute(
        'select record from images order by folder,source')]
    if any(r.get('image') is None or r['status'] == 'error' for r in records):
        raise ValueError('Resolve missing or corrupt source images before export')
    by_sha = defaultdict(list)
    for r in records:
        by_sha[r['sha256']].append(r)
    conflicts = {sha for sha, rs in by_sha.items() if len({r['folder_class_id'] for r in rs}) > 1}
    checked = json.loads((args.checked/'manifest.json').read_text())
    checked_by_sha = {r['source_sha256']: r for r in checked}
    backgrounds = json.loads(args.backgrounds.read_text())
    background_by_sha = {r['source_sha256']: r for r in backgrounds if r['accepted_background']}
    decisions = json.loads(args.decisions.read_text())
    rejected_ids = {r['id'] for r in decisions if r['status'].startswith('needs_')}
    rejected_sha = {r['sha256'] for r in records if r['id'] in rejected_ids}
    groups = leakage_groups(records, root, root.parent/'phash_cache.json')
    group_classes = defaultdict(set)
    for r in records:
        group_classes[groups[r['sha256']]].add(r['folder_class_id'])
    conflict_groups = {g for g, classes in group_classes.items() if len(classes) > 1}
    rejected_groups = {groups[sha] for sha in rejected_sha}
    candidates, excluded = {}, {}
    for sha, rs in by_sha.items():
        r = rs[0]
        reason = None
        if sha in conflicts:
            reason = 'identical_image_has_conflicting_source_classes'
        elif groups[sha] in conflict_groups:
            reason = 'near_duplicate_group_has_conflicting_source_classes'
        elif r['folder_class_id'] == 12 and sha in background_by_sha:
            candidates[sha] = {'record': r, 'objects': [],
                               'quality': 'ai_visually_screened_background',
                               'review': background_by_sha[sha]}
        elif r['folder_class_id'] >= 9:
            reason = 'healthy_image_uncertain_or_new_class_needs_semantic_annotation'
        elif sha in checked_by_sha:
            meta = checked_by_sha[sha]
            objects = read_polygons(args.checked/'labels'/(meta['id']+'.txt'))
            candidates[sha] = {'record': r, 'objects': objects,
                               'quality': 'ai_visually_screened', 'review': meta}
        elif groups[sha] in rejected_groups:
            reason = 'visual_review_found_problems_in_this_image_or_related_variant'
        else:
            second_path = args.ensemble/(sha+'.json')
            if not second_path.exists():
                raise ValueError('Second-model pass is incomplete: '+sha)
            second = json.loads(second_path.read_text())
            ok, reason = agreement(r['objects'], second, r['folder_class_id'])
            if ok:
                candidates[sha] = {'record': r, 'objects': r['objects'],
                                   'quality': 'two_model_pseudo_label', 'review': None}
                reason = None
        if reason:
            excluded[sha] = reason

    # Never treat a disagreement with a checked image's source class as truth.
    candidate_groups = defaultdict(list)
    for sha, c in candidates.items():
        candidate_groups[groups[sha]].append((sha, c))
    split = {}
    for class_id in list(range(9))+[12]:
        keys = [g for g, items in candidate_groups.items()
                if any(c['record']['folder_class_id'] == class_id for _, c in items)]
        keys.sort(key=lambda x: hashlib.sha256(('rice-v1:'+x).encode()).hexdigest())
        n_val = max(1, round(len(keys)*.2)) if len(keys) > 1 else 0
        for g in keys:
            proposed = 'val' if g in keys[:n_val] else 'train'
            # A near-duplicate group containing multiple classes stays one split.
            if g not in split or proposed == 'val':
                split[g] = proposed

    out.mkdir(parents=True)
    for name in ['images/pool', 'labels/pool', 'review/draft_labels',
                 'review/object_proposals', 'review/previews', 'weights']:
        (out/name).mkdir(parents=True)
    manifest = []
    train, val, checked_val = [], [], []
    counts = Counter()
    class_counts = defaultdict(Counter)
    for sha, rs in by_sha.items():
        canonical = rs[0]
        candidate = candidates.get(sha)
        for index, r in enumerate(rs):
            copy_file(root/r['image'], out/'images/pool'/(r['id']+'.jpg'))
            if r.get('draft_label'):
                copy_file(root/r['draft_label'], out/'review/draft_labels'/(r['id']+'.txt'))
            if r.get('proposal_file'):
                copy_file(root/r['proposal_file'], out/'review/object_proposals'/(r['id']+'.json'))
            copy_file(root/r['preview'], out/'review/previews'/(r['id']+'.jpg'))
            disposition = 'excluded'
            reason = excluded.get(sha)
            label_path = None
            if candidate:
                label_path = 'labels/pool/'+r['id']+'.txt'
                (out/label_path).write_text(yolo_text(candidate['objects']), encoding='utf-8')
                if index == 0:
                    disposition = split[groups[sha]]
                    image_line = './images/pool/'+r['id']+'.jpg'
                    (train if disposition == 'train' else val).append(image_line)
                    if disposition == 'val' and candidate['review']:
                        checked_val.append(image_line)
                    class_counts[r['folder_class_id']][disposition] += 1
                else:
                    disposition, reason = 'duplicate', 'exact_duplicate_not_repeated_in_training'
            counts[disposition] += 1
            manifest.append({
                'id': r['id'], 'source': r['source'], 'source_sha256': sha,
                'source_folder': r['folder'], 'source_class_id': r['folder_class_id'],
                'image': 'images/pool/'+r['id']+'.jpg', 'label': label_path,
                'disposition': disposition, 'reason': reason,
                'canonical_id': canonical['id'], 'split_group': groups[sha],
                'annotation_quality': candidate['quality'] if candidate else 'not_training_ready',
                'expert_diagnosis_confirmed': False,
                'visual_review': candidate['review'] if candidate and index == 0 else None,
            })
    if not train or not val:
        raise ValueError('No nonempty train/validation split')
    for name, lines in [('train.txt', train), ('val.txt', val), ('val_checked.txt', checked_val)]:
        (out/name).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'data.yaml').write_text(yaml_text('train.txt', 'val.txt'), encoding='utf-8')
    if checked_val:
        (out/'data_checked_val.yaml').write_text(yaml_text('train.txt', 'val_checked.txt'), encoding='utf-8')
    write_json(out/'manifest.json', manifest)
    positive_classes = sorted({c['record']['folder_class_id'] for c in candidates.values() if c['objects']})
    pseudo_count = sum(c['quality'] == 'two_model_pseudo_label' for c in candidates.values())
    report = {
        'source_images': len(records), 'unique_exact_contents': len(by_sha),
        'all_inputs_accounted_for': len(manifest) == len(records),
        'dispositions': dict(counts), 'source_class_conflict_groups': len(conflicts),
        'train_images': len(train), 'validation_images': len(val),
        'visually_screened_val_images': len(checked_val),
        'selected_annotation_quality': dict(Counter(c['quality'] for c in candidates.values())),
        'source_files_with_usable_annotations_including_duplicates': sum(bool(r['label']) for r in manifest),
        'class_counts': {str(k): dict(v) for k,v in class_counts.items()},
        'excluded_unique_reason_counts': dict(Counter(excluded.values())),
        'near_duplicate_grouping': 'exact SHA256 and 64bit pHash <=4 across flips/90deg rotations',
        'split_independence_limit': 'same-field/source relationships unknown; not a leakage guarantee',
        'has_unreviewed_pseudo_labels': bool(pseudo_count),
        'validation_reference': ('includes unreviewed pseudo-labels' if pseudo_count else
                                 'AI visually screened masks/backgrounds; not expert ground truth'),
        'positive_classes_present': positive_classes,
        'missing_positive_classes': sorted(set(range(9))-set(positive_classes)),
        'training_scope': 'additional data to combine with original annotated dataset',
        'new_class_semantic_annotations_completed': False,
        'background_policy': 'visually screened Healthy source images only; empty labels; no new class',
        'all_images_training_ready': not excluded,
        'model_sha256': digest_file(args.model),
        'second_model': json.loads((args.ensemble/'config.json').read_text()),
    }
    write_json(out/'report.json', report)
    copy_file(args.model, out/'weights/best.pt')
    for name in ['handoff_validate.py', 'merge_rice_training.py']:
        shutil.copyfile(Path(__file__).parent/name, out/name)
    shutil.copyfile(Path(__file__).parent/'RICE_HANDOFF_README.md', out/'README_TH.md')
    (out/'requirements.txt').write_text('ultralytics==8.4.137\n', encoding='utf-8')
    with (out/'excluded.csv').open('w', encoding='utf-8-sig', newline='') as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=['id','source','source_folder','reason'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in manifest if r['disposition']=='excluded')
    print(json.dumps(report, indent=2))


def package(out):
    out = out.resolve()
    zip_path = out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', allowZip64=True) as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix not in {'.cache', '.pyc'} and '__pycache__' not in p.parts:
                archive.write(p, Path(out.name)/p.relative_to(out),
                              compress_type=zipfile.ZIP_STORED if p.suffix in {'.jpg','.pt'} else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(zip_path) as archive:
        error = archive.testzip()
        if error:
            raise ValueError('ZIP CRC failure: '+error)
    print(json.dumps({'zip': str(zip_path), 'bytes': zip_path.stat().st_size,
                      'sha256': digest_file(zip_path)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('temp/rice_labeling/local_all'))
    p.add_argument('--ensemble', type=Path, default=Path('temp/rice_labeling/ensemble'))
    p.add_argument('--checked', type=Path, default=Path('temp/rice_labeling/visually_checked_yolo'))
    p.add_argument('--decisions', type=Path, default=Path('temp/rice_labeling/visual_audit/review_decisions.json'))
    p.add_argument('--backgrounds', type=Path, default=Path('temp/rice_labeling/new_class_audit/background_review.json'))
    p.add_argument('--model', type=Path, default=Path('best.pt'))
    p.add_argument('--output', type=Path, default=Path('temp/rice_labeling/rice_training_handoff'))
    p.add_argument('--package-only', action='store_true')
    a = p.parse_args()
    if not a.package_only:
        build(a)
    package(a.output)
