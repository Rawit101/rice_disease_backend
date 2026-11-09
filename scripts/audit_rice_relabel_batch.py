"""Audit old rice masks and propose conservative, class-preserving refinements."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from publish_rice_class_names import BASE, NAMES, parse_label


WORK = BASE / 'rice_relabel_work'
INPUT = WORK / 'inventory.json'


def read_old(row):
    if not row['old_label']:
        return [], None
    raw = (BASE / row['old_label']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['label_sha256']:
        raise RuntimeError(f"Source label changed for {row['id']}")
    return parse_label(raw.decode('utf-8'), NAMES), raw


def load_rgb(row, limit=560):
    with Image.open(BASE / row['source_image']) as image:
        image.draft('RGB', (limit, limit))
        image = image.convert('RGB')
        image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        return np.asarray(image)


def shape_mask(polygon, shape):
    h, w = shape
    points = np.rint(np.asarray(polygon) * [w - 1, h - 1]).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def contour_polygon(contour, shape):
    h, w = shape
    curve = cv2.approxPolyDP(contour, max(0.65, cv2.arcLength(contour, True) * 0.003), True)
    pts = curve.reshape(-1, 2)
    if len(pts) < 3 or len(set(map(tuple, pts))) != len(pts) or cv2.contourArea(pts) < 5:
        return None
    return (pts / [w - 1, h - 1]).round(7).tolist()


def features(rgb, cls):
    h, s, v = np.moveaxis(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), -1, 0)
    red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
    healthy = (h >= 38) & (h <= 93) & (s >= 60) & (v >= 42) & (green >= red + 8)
    blank = ((s <= 17) & (v >= 241)) | (v <= 20)
    if cls in {0, 9}:
        support = (h >= 9) & (h <= 35) & (s >= 59) & (v >= 66) & (red >= blue + 24)
        core = support & (s >= 92) & (red >= green + 7)
    else:
        support = ((h <= 31) | (h >= 173)) & (s >= 48) & (v >= 38) & (red >= green + 7)
        core = support & (red >= green + 12) & (red >= blue + 18)
    return healthy, blank, support, core


def refine_mask(mask, support, core, healthy, cls):
    area = int(mask.sum())
    if area < 35:
        return None
    inside = mask.astype(bool)
    if np.count_nonzero(inside & healthy) < area * .17:
        return None
    pieces, _ = cv2.findContours((inside & support).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not pieces:
        return None
    pieces = sorted(pieces, key=cv2.contourArea, reverse=True)
    leading = cv2.contourArea(pieces[0])
    if leading < max(10, area * .18) or leading > area * .78:
        return None
    if len(pieces) > 1 and cv2.contourArea(pieces[1]) > leading * .23:
        return None
    polygon = contour_polygon(pieces[0], mask.shape)
    if polygon is None:
        return None
    tight = shape_mask(polygon, mask.shape).astype(bool)
    if np.count_nonzero(tight & core) < tight.sum() * .43:
        return None
    if np.count_nonzero(tight & ~inside) > tight.sum() * .06:
        return None
    if np.count_nonzero(tight & healthy) > tight.sum() * .09:
        return None
    if cls in {0, 9}:
        (_, _), (a, b), _ = cv2.minAreaRect(pieces[0])
        if min(a, b) < 2 or max(a, b) / min(a, b) < 2.4:
            return None
    return polygon


def audit(row):
    flags = set()
    try:
        objects, _ = read_old(row)
    except (ValueError, UnicodeError):
        objects = []
        flags.add('invalid_old_label')
    ids = sorted({o['class_id'] for o in objects})
    cls = row['expected_class_id']
    if not objects:
        flags.add('missing_usable_label')
    if objects and ids != [cls]:
        flags.add('old_class_disagrees_with_source')
    if row['source_class_conflict']:
        flags.add('source_class_conflict')
    if row['photo_decision'] == 'hold_visual_unclear':
        flags.add('photo_previously_unclear')
    result = dict(flags=[], old_count=len(objects), old_class_ids=ids)
    try:
        rgb = load_rgb(row)
    except (OSError, ValueError):
        result.update(flags=sorted(flags | {'image_unreadable'}))
        return row['id'], result
    if not objects:
        result['flags'] = sorted(flags)
        return row['id'], result
    healthy, blank, support, core = features(rgb, cls)
    shape = rgb.shape[:2]
    union = np.zeros(shape, np.uint8)
    refined = []
    changed = 0
    # A stored mask is a localization seed, with its original class retained.
    eligible = (cls in {0, 1, 2, 3, 9} and not flags and
                row['photo_decision'] in {'reviewed_keep', 'auto_candidate'} and len(objects) <= 18)
    for obj in objects:
        polygon = obj['polygon']
        points = np.asarray(polygon, np.float32)
        if len(set(map(tuple, polygon))) != len(polygon):
            flags.add('repeated_vertices')
        if cv2.contourArea(points) < 1e-8:
            flags.add('zero_or_tiny_polygon')
        if np.any(points <= .0001) or np.any(points >= .9999):
            flags.add('polygon_touches_edge')
        mask = shape_mask(polygon, shape)
        union |= mask
        if eligible:
            count = int(mask.sum())
            tight = refine_mask(mask, support, core, healthy, cls)
            if tight:
                refined.append(tight)
                changed += 1
            elif count and np.count_nonzero(mask & support) >= count * .73:
                refined.append(polygon)
    coverage = int(union.sum())
    background = np.count_nonzero(union & blank) / max(coverage, 1)
    green = np.count_nonzero(union & healthy) / max(coverage, 1)
    if coverage > union.size * .5:
        flags.add('mask_covers_half_image')
    if background >= .22 and coverage >= 15:
        flags.add('mask_includes_uniform_background')
    if cls in {0, 1, 2, 3, 8, 9} and green >= .44 and coverage >= 15:
        flags.add('mask_contains_substantial_green_tissue')
    if coverage <= 3:
        flags.add('mask_too_small_to_review_at_audit_scale')
    result.update(flags=sorted(flags), mask_fraction=round(coverage / union.size, 5),
                  background_fraction=round(background, 4), green_fraction=round(green, 4))
    unsafe = {'repeated_vertices', 'zero_or_tiny_polygon', 'mask_includes_uniform_background'}
    if changed and len(refined) == len(objects) and not unsafe.intersection(flags):
        result.update(proposed_polygons=refined, changed_objects=changed)
    return row['id'], result


def run():
    cv2.setNumThreads(1)
    rows = json.loads(INPUT.read_text(encoding='utf-8'))
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (image_id, result) in enumerate(pool.map(audit, rows.values()), 1):
            results[image_id] = result
            if i % 1000 == 0:
                print(f'Audited {i:,} / {len(rows):,}', flush=True)
    path = WORK / 'batch_audit.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(results, separators=(',', ':')) + '\n', encoding='utf-8')
    temp.replace(path)
    print('Flags', dict(Counter(f for r in results.values() for f in r['flags'])), flush=True)
    print('Conservative polygon proposals', sum('proposed_polygons' in r for r in results.values()), flush=True)


if __name__ == '__main__':
    run()
