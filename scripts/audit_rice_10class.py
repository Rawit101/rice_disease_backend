"""Audit scope, label conflicts and geometry; never infer a disease from a folder.

The source category is an unverified hint used only to enforce requested scope
and flag disagreements. Passing these checks is not semantic ground truth.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse, csv, hashlib, json, math, random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from publish_rice_class_names import NAMES, THAI, CLASSES, parse_label

BASE = Path(__file__).resolve().parents[1] / 'temp/rice_labeling'
OUT = BASE / 'rice_10class_audit'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')


def allowed_class(folder, mapping):
    """Bakanae's catalog ID 9 must never be confused with export Tungro ID 9."""
    name = mapping.get(folder)
    return NAMES.index(name) if name in NAMES else None


def inventory():
    mapping = read(BASE / 'rice_archive_corrected/class_mapping.json')['source_folder_to_name']
    additions = {r['id']: r for r in read(BASE / 'rice_tungro_labeled_v2/manifest.json')}
    manual = {r['id']: r for r in read(BASE / 'remaining_recheck/reviewed_additions/manifest.json')}
    rows, excluded, categories = [], [], defaultdict(set)
    for dataset in ['rice_full_auto', 'rice_archive_corrected']:
        for original in read(BASE / dataset / 'manifest.json'):
            r = dict(original)
            categories[r['sha256']].add(mapping.get(r['source_folder'], r['source_folder']))
            cls = allowed_class(r['source_folder'], mapping)
            if cls is None:
                excluded.append(dict(id=r['id'], source_folder=r['source_folder'],
                                     source_category=mapping.get(r['source_folder']),
                                     reason='outside_requested_10_classes', had_label=bool(r['label'])))
                continue
            root = BASE / dataset
            if r['id'] in additions:
                r = dict(additions[r['id']]); root = BASE / 'rice_tungro_labeled_v2'
                dataset_name = 'rice_tungro_labeled_v2'
            elif r['id'] in manual:
                root = BASE / 'remaining_recheck/reviewed_additions'
                r.update(image='images/' + r['id'] + '.jpg', label='labels/' + r['id'] + '.txt',
                         quality='ai_visually_reviewed_manual_addition')
                dataset_name = 'remaining_recheck'
            else:
                dataset_name = dataset
            rows.append(dict(r, dataset=dataset_name, dataset_root=str(root.relative_to(BASE)),
                             expected_class_id=cls, expected_class_name=NAMES[cls]))
    for r in rows:
        r['exact_content_source_categories'] = sorted(categories[r['sha256']])
    return rows, excluded


def geometry(objects):
    flags, areas = set(), []
    for o in objects:
        p = np.asarray(o['polygon'], dtype=float)
        area = abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))) / 2
        areas.append(area)
        if area <= 1e-8: flags.add('degenerate_or_tiny_polygon')
        if area > .65: flags.add('very_large_mask_requires_target_review')
        if len(set(map(tuple, p))) != len(p): flags.add('repeated_vertices_require_topology_review')
        if np.any((p < .001) | (p > .999)): flags.add('mask_touches_image_border')
    return flags, areas


def one(r):
    root = BASE / r['dataset_root']
    flags = set()
    source_reasons = set(r.get('reasons', []))
    # Metadata conflicts are review signals, never proof that the folder is right.
    if len(r['exact_content_source_categories']) > 1: flags.add('exact_image_has_conflicting_source_categories')
    if 'near_duplicate_group_has_conflicting_source_classes' in source_reasons: flags.add('near_duplicate_source_category_conflict')
    if 'visual_review_previously_found_mask_problems' in source_reasons: flags.add('previous_visual_mask_problem')
    if 'models_disagree_on_overlapping_object_class' in source_reasons: flags.add('teacher_class_disagreement')
    if 'low_confidence_high_resolution_recovery' in source_reasons: flags.add('low_confidence_recovery')
    if r.get('has_unreviewed_boundary_proposal'): flags.add('previous_boundary_proposal_unaccepted')
    objects, label_sha = [], None
    if r.get('label'):
        raw = (root / r['label']).read_bytes()
        label_sha = hashlib.sha256(raw).hexdigest()
        try: objects = parse_label(raw.decode(), NAMES)
        except (ValueError, IndexError): flags.add('invalid_label_geometry_or_class')
    if not objects: flags.add('no_usable_mask_not_a_negative')
    ids = sorted(set(o['class_id'] for o in objects))
    if objects and ids != [r['expected_class_id']]: flags.add('annotation_class_disagrees_with_source_hint')
    if objects and r['expected_class_id'] not in ids: flags.add('no_mask_for_source_hint_class')
    if len(ids) > 1: flags.add('multiple_disease_classes_require_confirmation')
    if r['expected_class_id'] == 9:
        flags.add('tungro_diagnosis_not_established_by_color_or_sam')
        if r['quality'] == 'automatic_symptom_mask_draft': flags.add('tungro_mask_not_individually_reviewed')
    geom, areas = geometry(objects); flags |= geom
    try:
        with Image.open(root / r['image']) as source:
            im = source.convert('RGB'); width, height = im.size; im.thumbnail((640, 640))
            rgb = np.asarray(im)
        if min(width, height) < 160: flags.add('low_source_resolution')
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for o in objects:
            points = np.rint(np.asarray(o['polygon']) * [im.width-1, im.height-1]).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Flag obvious near-white/near-black backdrop overlap. It is not a leaf classifier.
        backdrop = ((hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 247)) | (hsv[:, :, 2] < 8)
        background_fraction = float(np.count_nonzero(backdrop & (mask > 0)) / max(1, mask.sum()))
        mask_fraction = float(mask.mean())
        if objects and background_fraction > .25: flags.add('mask_overlaps_near_uniform_background')
        if objects and mask_fraction > .65: flags.add('most_image_is_masked_requires_target_review')
    except Exception as error:
        width = height = 0; background_fraction = mask_fraction = None
        flags.add('image_decode_error'); r['image_error'] = str(error)
    return dict(r, class_ids=ids, object_count=len(objects), label_sha256=label_sha,
                audit_flags=sorted(flags), width=width, height=height,
                background_overlap_fraction=background_fraction, mask_image_fraction=mask_fraction,
                max_polygon_area=max(areas, default=0),
                class_hint_is_ground_truth=False, training_ready_after_automatic_audit=False)


def scan():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, excluded = inventory()
    output = []
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(one, rows), 1):
            output.append(result)
            if i % 1000 == 0: print(json.dumps(dict(stage='ten_class_audit', completed=i, total=len(rows))), flush=True)
    assert len({r['id'] for r in output}) == len(output)
    write(OUT / 'audit_manifest.json', output)
    write(OUT / 'excluded_outside_scope.json', excluded)
    write(OUT / 'classes.json', CLASSES)
    with (OUT / 'excluded_outside_scope.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, ['id', 'source_folder', 'source_category', 'reason', 'had_label']); w.writeheader(); w.writerows(excluded)
    flags = Counter(flag for r in output for flag in r['audit_flags'])
    report = dict(all_source_images=len(rows)+len(excluded), in_scope_images=len(output),
                  excluded_source_files=len(excluded), excluded_files_with_previous_labels=sum(r['had_label'] for r in excluded),
                  files_with_masks=sum(bool(r['object_count']) for r in output),
                  without_usable_masks=sum(not r['object_count'] for r in output),
                  flags=dict(flags), source_files_by_requested_class=dict(Counter(r['expected_class_name'] for r in output)),
                  diagnoses_verified_by_audit=False, training_executed=False,
                  source_scope_not_a_diagnosis=True, prior_exports_not_modified=True)
    write(OUT / 'audit_report.json', report)
    print(json.dumps(dict(stage='ten_class_scan_complete', **report)), flush=True)


def sheets():
    rows = read(OUT / 'audit_manifest.json')
    rng = random.Random(20261101)
    selected, seen = [], set()
    for cls in range(10):
        pool = [r for r in rows if r['expected_class_id'] == cls and r['object_count']]
        rng.shuffle(pool)
        chosen = []
        groups = [
            [r for r in pool if 'annotation_class_disagrees_with_source_hint' in r['audit_flags']],
            [r for r in pool if 'mask_overlaps_near_uniform_background' in r['audit_flags']],
            [r for r in pool if 'annotation_class_disagrees_with_source_hint' not in r['audit_flags']],
            [r for r in pool if r['quality'].startswith('ai_visually')], pool]
        # Balance risk examples with class-matching examples, deduplicating file contents.
        for group, limit in zip(groups, [3, 2, 5, 2, 16]):
            count = 0
            for r in group:
                if len(chosen) >= (24 if cls == 9 else 12): break
                if r['sha256'] in seen: continue
                chosen.append(r); seen.add(r['sha256']); count += 1
                if count >= limit: break
        selected.extend(chosen)
    # Include exactly the example currently shown by the user and a known cross-class case.
    for rid in ['189bc108ac882ec2ad8b', 'd0166523e30efd80eaef', 'cb30bbadf1d93822561e']:
        r = next(r for r in rows if r['id'] == rid)
        if r['id'] not in {x['id'] for x in selected}: selected.append(r)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 17)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 14)
    dest = OUT / 'visual_sheets'; dest.mkdir(exist_ok=True)
    entries = []
    for start in range(0, len(selected), 6):
        page = Image.new('RGB', (1280, 1110), 'white'); draw = ImageDraw.Draw(page)
        for j, r in enumerate(selected[start:start+6]):
            index = start + j + 1; x = j % 2 * 640; y = j // 2 * 370
            root = BASE / r['dataset_root']; objects = parse_label((root/r['label']).read_text(), NAMES)
            with Image.open(root / r['image']) as source:
                im = source.convert('RGB'); im.thumbnail((310, 290))
            overlay = im.copy().convert('RGBA'); layer = Image.new('RGBA', im.size)
            pen = ImageDraw.Draw(layer)
            for o in objects:
                color = [(255,190,20,70),(30,240,190,70),(255,85,150,70),(90,150,255,70)][o['class_id']%4]
                points = [(a*im.width,b*im.height) for a,b in o['polygon']]
                pen.polygon(points, fill=color, outline=color[:3]+(255,), width=2)
            overlay = Image.alpha_composite(overlay, layer).convert('RGB')
            page.paste(im, (x+(310-im.width)//2, y+45+(290-im.height)//2))
            page.paste(overlay, (x+320+(310-im.width)//2, y+45+(290-im.height)//2))
            draw.text((x+5,y+2), '%03d hint=%d %s | masks=%s' % (index,r['expected_class_id'],NAMES[r['expected_class_id']],','.join(map(str,r['class_ids']))), font=font, fill='black')
            draw.text((x+5,y+24), r['id']+' | '+Path(r['source']).name[:43], font=small, fill='black')
            draw.text((x+5,y+338), 'BG %.2f  coverage %.2f | %s' % (r['background_overlap_fraction'],r['mask_image_fraction'],r['quality'][:40]), font=small, fill='black')
            entries.append(dict(index=index,id=r['id'],sheet='visual_sheets/page_%02d.jpg'%(start//6+1),
                                expected_class_id=r['expected_class_id'],class_ids=r['class_ids'],audit_flags=r['audit_flags']))
        page.save(dest / ('page_%02d.jpg' % (start//6+1)), quality=93)
    write(OUT / 'visual_selection.json', entries)
    print(json.dumps(dict(stage='visual_sheets_ready', images=len(entries), pages=math.ceil(len(entries)/6))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['scan','sheets']); args = parser.parse_args()
    scan() if args.action == 'scan' else sheets()
