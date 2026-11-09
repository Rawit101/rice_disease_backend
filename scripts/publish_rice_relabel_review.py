"""Publish reviewed YOLO drafts and the disposition of every audited rice photo."""
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_rice_relabel_batch import BASE, INPUT, WORK, read_old
from publish_rice_class_names import CLASSES, NAMES, parse_label, write_csv


NAME = 'rice_relabel_review'
STATUS_TH = {
    'redrawn_draft': 'วาดขอบเขตรอยอาการใหม่ รอยืนยันชื่อโรค',
    'retained_draft': 'คงขอบเขตเดิมหลังเปิดเทียบ รอยืนยันชื่อโรค',
    'held_after_visual_review': 'พักหลังเปิดเทียบภาพกับ label',
    'held_source_class_conflict': 'พักเพราะภาพที่จับคู่ได้อยู่คนละหมวดโรค',
    'held_old_class_conflict': 'พักเพราะคลาสเดิมต่างจากหมวดต้นทาง',
    'held_missing_label': 'พักเพราะไม่มี label เดิมที่ใช้ได้',
    'held_boundary_risk': 'พักเพราะพบความเสี่ยงด้านขอบเขต',
    'held_photo_or_review_pending': 'พักเพราะคุณภาพภาพหรือขอบเขตยังรอการเปิดตรวจ',
}
BOUNDARY_FLAGS = {'repeated_vertices', 'zero_or_tiny_polygon', 'mask_covers_half_image',
                  'mask_includes_uniform_background', 'mask_contains_substantial_green_tissue',
                  'mask_too_small_to_review_at_audit_scale'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')


def simple_polygon(poly):
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != 2 or not np.isfinite(p).all():
        return False
    if p.min() < 0 or p.max() > 1 or len(set(map(tuple, p))) != len(p):
        return False
    if cv2.contourArea(p.astype(np.float32)) <= 1e-8:
        return False
    q = np.roll(p, -1, axis=0)
    for i in range(len(p) - 2):
        j = slice(i + 2, len(p) - (i == 0))
        a, b, c, d = p[i], q[i], p[j], q[j]
        eps = 1e-12
        overlaps = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                          np.minimum(np.maximum(a, b), np.maximum(c, d)) + eps, axis=1)
        ab_c, ab_d = np.cross(b - a, c - a), np.cross(b - a, d - a)
        cd_a, cd_b = np.cross(d - c, a - c), np.cross(d - c, b - c)
        spans_ab = (np.minimum(ab_c, ab_d) <= eps) & (np.maximum(ab_c, ab_d) >= -eps)
        spans_cd = (np.minimum(cd_a, cd_b) <= eps) & (np.maximum(cd_a, cd_b) >= -eps)
        if np.any(overlaps & spans_ab & spans_cd):
            return False
    return True


def label_for(row, visual):
    if not visual or visual['status'] == 'hold' or row['source_class_conflict']:
        return None, None
    if visual['status'] == 'keep':
        objects, label = read_old(row)
        if not objects or {obj['class_id'] for obj in objects} != {row['expected_class_id']}:
            return None, 'Current label is missing or has a different class.'
    else:
        assert visual['status'] == 'redo'
        lines = []
        for polygon in visual['polygons']:
            coordinates = ' '.join(f'{xy:.7f}' for point in polygon for xy in point)
            lines.append(f"{row['expected_class_id']} {coordinates}")
        label = ('\n'.join(lines) + '\n').encode('utf-8')
        objects = parse_label(label.decode('utf-8'), NAMES)
    if not objects or not all(simple_polygon(obj['polygon']) for obj in objects):
        return None, 'Polygon is degenerate, repeated or self-intersecting.'
    return label, None


def pending_status(flags):
    for flag, status in (
        ('source_class_conflict', 'held_source_class_conflict'),
        ('old_class_disagrees_with_source', 'held_old_class_conflict'),
        ('missing_usable_label', 'held_missing_label'),
    ):
        if flag in flags:
            return status
    if BOUNDARY_FLAGS.intersection(flags):
        return 'held_boundary_risk'
    return 'held_photo_or_review_pending'


def collect(inventory, audit):
    primary = {image_id for image_id, r in inventory.items() if r['photo_decision'] == 'reviewed_keep'}
    visual = {}
    for name in ('main', 'rare', 'spot', 'tungro'):
        decisions = load(WORK / f'{name}.json')
        assert decisions.keys() <= primary and not visual.keys() & decisions.keys(), name
        visual.update(decisions)
    assert visual.keys() == primary, f'Missing primary review {primary - visual.keys()}'
    for name in ('main_held', 'rare_held', 'spot_held', 'tungro_held', 'auto_visual'):
        path = WORK / f'{name}.json'
        if path.exists():
            for image_id, decision in load(path).items():
                assert image_id in inventory
                visual.setdefault(image_id, decision)
    rows, drafts, dropped = [], {}, {}
    for image_id, row in inventory.items():
        decision = visual.get(image_id)
        checked = audit[image_id]
        flags = checked['flags']
        note = decision['note'] if decision else ''
        label, error = label_for(row, decision)
        if error:
            dropped[image_id] = error
            note += ' ' + error
        if label:
            drafts[image_id] = label
            status = 'redrawn_draft' if decision['status'] == 'redo' else 'retained_draft'
        elif row['source_class_conflict']:
            status = 'held_source_class_conflict'
        elif decision:
            status = 'held_after_visual_review'
        else:
            status = pending_status(flags)
        rows.append(dict(
            id=image_id, filename=row['filename'], group='selected_353' if image_id in primary else 'held_12211',
            source_folder=row['source_folder'], class_hint_id=row['expected_class_id'],
            class_hint_th=row['source_hint_th'], class_hint_en=row['expected_class_name'],
            photo_screen=row['photo_decision'], status=status, status_th=STATUS_TH[status],
            opened_to_compare=bool(decision), visual_note=note,
            prior_class_ids=' | '.join(map(str, checked['old_class_ids'])),
            prior_class_names=' | '.join(NAMES[i] for i in checked['old_class_ids']),
            prior_object_count=checked['old_count'], prior_label=row['old_label'] or '',
            audit_flags=' | '.join(flags), prior_audit_flags=' | '.join(row['audit_flags']),
            prior_mask_fraction=checked.get('mask_fraction', ''),
            prior_background_fraction=checked.get('background_fraction', ''),
            prior_green_fraction=checked.get('green_fraction', ''),
            source_image=row['source_image'], packaged_image=f'images/{image_id}.jpg' if label else '',
            packaged_label=f'draft_labels/{image_id}.txt' if label else '',
            draft_object_count=len(label.splitlines()) if label else 0,
            disease_confirmed=False, training_ready=False,
        ))
    return rows, drafts, dropped


def comparison_sheets(root, rows, inventory, drafts):
    directory = root / 'before_after'
    directory.mkdir()
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    for start in range(0, len(rows), 9):
        sheet = Image.new('RGB', (1440, 768), '#f3f5ed')
        titles = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows[start:start + 9]):
            image_id = row['id']
            x, y = i % 3 * 480, i // 3 * 256
            with Image.open(root / row['packaged_image']) as original:
                photo = original.convert('RGB')
                photo.thumbnail((230, 220), Image.Resampling.LANCZOS)
            try:
                old, _ = read_old(inventory[image_id])
            except ValueError:
                old = []
            new = parse_label(drafts[image_id].decode('utf-8'), NAMES)
            for index, (objects, color) in enumerate(((old, '#fd3369'), (new, '#08e7fb'))):
                overlay = photo.copy()
                draw = ImageDraw.Draw(overlay)
                for obj in objects:
                    points = [(xx * (photo.width - 1), yy * (photo.height - 1))
                              for xx, yy in obj['polygon']]
                    draw.line(points + [points[0]], fill=color, width=2)
                sheet.paste(overlay, (x + 240 * index + (240 - photo.width) // 2, y + 33))
            titles.text((x + 4, y + 3), f'{image_id} | {row["class_hint_en"]}', fill='#19241c', font=font)
            titles.text((x + 4, y + 18), 'Old (pink)', fill='#862742', font=font)
            titles.text((x + 244, y + 18), 'New draft (cyan)', fill='#05656c', font=font)
        sheet.save(directory / f'page_{start // 9 + 1:02d}.jpg', quality=94)


def publish():
    inventory = load(INPUT)
    audit = load(WORK / 'batch_audit.json')
    assert len(inventory) == 12564 and set(audit) == set(inventory)
    rows, drafts, dropped = collect(inventory, audit)
    ready = [r for r in rows if r['packaged_label']]
    assert len({inventory[r['id']]['pixel_sha256'] for r in ready}) == len(ready)
    report = dict(
        audited=len(rows), selected_group=353, held_group=12211,
        opened_to_compare=sum(r['opened_to_compare'] for r in rows),
        opened_from_held=sum(r['opened_to_compare'] and r['group'] == 'held_12211' for r in rows),
        status_counts=dict(Counter(r['status'] for r in rows)),
        drafts_by_group=dict(Counter(r['group'] for r in ready)),
        draft_counts_by_class={
            name: dict(Counter(r['status'] for r in ready if r['class_hint_en'] == name))
            for name in NAMES
        },
        automatic_proposals_tested=sum('proposed_polygons' in result for result in audit.values()),
        automatic_proposals_exported=0, topology_holds=dropped,
        audit_flag_counts=dict(Counter(f for r in audit.values() for f in r['flags'])),
        verified_training_images=0,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    folder, archive_path = BASE / NAME, BASE / f'{NAME}.zip'
    if folder.exists() or archive_path.exists():
        raise FileExistsError(folder)
    with tempfile.TemporaryDirectory(dir=BASE) as scratch:
        root = Path(scratch) / NAME
        root.mkdir()
        for name in ('images', 'draft_labels'):
            (root / name).mkdir()
        for row in ready:
            shutil.copyfile(BASE / row['source_image'], root / row['packaged_image'])
            (root / row['packaged_label']).write_bytes(drafts[row['id']])
        for name, subset in [('all_review_results.csv', rows), ('draft_index.csv', ready)]:
            write_csv(root / name, list(rows[0]), subset)
        write_csv(root / 'classes.csv', list(CLASSES[0]), CLASSES)
        write_json(root / 'summary.json', report)
        changed = [r for r in ready if r['status'] == 'redrawn_draft']
        comparison_sheets(root, changed, inventory, drafts)
        retained = len(ready) - len(changed)
        (root / 'README_TH.txt').write_text(
            'ผลตรวจและแก้ label โรคข้าวแบบล้อมเฉพาะบริเวณอาการ\n\n'
            f"ตรวจอัตโนมัติครบ {len(rows):,} ไฟล์จากชุดคัด 353 ภาพและกลุ่มพัก 12,211 ไฟล์\n"
            f"เปิดเทียบภาพและ label {report['opened_to_compare']:,} ภาพ "
            f"รวมจากกลุ่มพัก {report['opened_from_held']:,} ภาพ\n"
            f"วาดใหม่ {len(changed):,} ภาพ และคง label เดิมหลังตรวจ {retained:,} ภาพ\n\n"
            'เปิด before_after เพื่อเทียบกรอบเดิมสีชมพูกับร่างใหม่สีฟ้า\n'
            'images และ draft_labels จับคู่ด้วยชื่อไฟล์เดียวกัน ข้อมูลเป็น YOLO segmentation\n'
            'draft_index.csv เชื่อมภาพกับชื่อไฟล์ต้นทาง ชื่อคลาส และเหตุผลรายภาพ\n'
            'all_review_results.csv รวมผลทุกไฟล์ที่ตรวจ ช่อง opened_to_compare ระบุการเปิดเทียบรายภาพ\n'
            'คลาสอ้างอิงจากรูปแบบอาการที่มองเห็นประกอบกับข้อมูลต้นทาง รอการยืนยันก่อนเทรน\n'
            'ภาพพักไม่มีไฟล์ label ใน ZIP ให้ใช้รายการเหตุผลเพื่อไล่ตรวจต่อ\n\n'
            'คู่มืออาการจาก IRRI\n'
            'https://www.knowledgebank.irri.org/images/docs/diagnostic-of-common-diseases-of-rice.pdf\n',
            encoding='utf-8-sig',
        )
        packed = Path(scratch) / archive_path.name
        files = [p for p in sorted(root.rglob('*')) if p.is_file()]
        with zipfile.ZipFile(packed, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
            for path in files:
                mode = zipfile.ZIP_STORED if path.suffix == '.jpg' else zipfile.ZIP_DEFLATED
                z.write(path, str(path.relative_to(root.parent)), compress_type=mode)
        with zipfile.ZipFile(packed) as z:
            assert z.testzip() is None and len(z.namelist()) == len(files)
        root.rename(folder)
        packed.rename(archive_path)
    print(archive_path, flush=True)


if __name__ == '__main__':
    publish()
