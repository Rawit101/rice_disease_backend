"""Combine file, pixel, and visual screening into a photo-only handoff."""
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from publish_rice_class_names import BASE, CLASSES, write_csv


WORK = BASE / 'rice_image_quality'
NAME = 'rice_photo_selection_complete'
STATUS_TH = {
    'reviewed_keep': 'คัดไว้หลังเปิดดูภาพ',
    'auto_candidate': 'ผ่านเกณฑ์ภาพอัตโนมัติ รอเปิดตรวจ',
    'hold_class_conflict': 'พักเพราะภาพเหมือนกันแต่ชื่อโรคต้นทางขัดกัน',
    'hold_visual_unclear': 'พักเพราะอาการไม่ชัดเมื่อเปิดดูภาพ',
    'hold_automatic_uncertain': 'พักเพราะหลักฐานจากการคัดอัตโนมัติยังไม่พอ',
    'omit_quality': 'ตัดออกเพราะคุณภาพภาพ',
    'near_duplicate_visual': 'ตัดภาพใกล้ซ้ำจากการเปิดดู',
    'near_duplicate_aligned': 'ตัดภาพใกล้ซ้ำจากการจัดแนวพิกเซล',
    'exact_duplicate': 'ตัดสำเนาที่พิกเซลเหมือนกันทุกจุด',
}
VISUAL_STATUS = {
    'K': 'reviewed_keep', 'H': 'hold_visual_unclear',
    'O': 'omit_quality', 'D': 'near_duplicate_visual',
}
AUTO_STATUS = {
    'A': 'auto_candidate', 'H': 'hold_automatic_uncertain',
    'O': 'omit_quality', 'D': 'near_duplicate_aligned',
}
CANDIDATE_PRIORITY = ('reviewed_keep', 'auto_candidate')


def read(filename):
    return json.loads((WORK / filename).read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def collect_decisions():
    rows = read('quality_metrics.json')
    by_id = {row['id']: row for row in rows}
    assert len(rows) == len(by_id) and all(row['image_decode_ok'] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['pixel_sha256']].append(row)
    first = {row['id']: copies[0]['id'] for copies in groups.values() for row in copies}
    automatic = read('full_auto_triage.json')
    assert set(automatic) == set(first.values()), 'Automatic screening must cover each pixel group'

    visual = {}
    for row in read('photo_inspection_manifest.json'):
        visual[first[row['id']]] = (row['id'], row['decision'], row['note'])
    for name in ('common', 'rare', 'tungro', 'auto_qa'):
        filename = f'full_review_{name}.json'
        if name == 'auto_qa' and not (WORK / filename).exists():
            continue
        for image_id, (code, note) in read(filename).items():
            visual.setdefault(first[image_id], (image_id, code, note))

    conflicts = read('full_auto_class_conflicts.json')
    exact_classes = {
        copies[0]['id']: {row['expected_class_id'] for row in copies}
        for copies in groups.values()
    }
    decisions = {}
    similar = defaultdict(dict)

    def link(image_id, match_id, evidence):
        assert image_id != match_id and exact_classes[image_id] & exact_classes[match_id]
        similar[image_id].setdefault(match_id, evidence)
        similar[match_id].setdefault(image_id, evidence)

    for image_id, auto in automatic.items():
        viewed_id, code, note = visual.get(image_id, ('', auto['decision'], auto['note']))
        status = (VISUAL_STATUS if viewed_id else AUTO_STATUS)[code]
        match_id = auto.get('representative_id', '')
        if match_id:
            link(image_id, match_id, auto['alignment_evidence'])
        matches = conflicts.get(image_id, [])
        conflict = len(exact_classes[image_id]) > 1 or bool(matches or auto.get('source_class_conflict'))
        if conflict and status not in {'omit_quality', 'near_duplicate_visual', 'near_duplicate_aligned'}:
            status = 'hold_class_conflict'
        decisions[image_id] = dict(
            image_id=viewed_id or image_id, status=status, note=note, visually_reviewed=bool(viewed_id),
            visual_decision=code if viewed_id else '',
            source_class_conflict=conflict, auto_decision=auto['decision'],
            conflict_match_ids=' | '.join(match['other_id'] for match in matches),
            aligned_match_id=match_id if status == 'near_duplicate_aligned' else '',
            alignment_evidence=auto.get('alignment_evidence', {}) if status == 'near_duplicate_aligned' else {},
        )

    for image_id, evidence in read('tungro_verified_variant_candidates.json').items():
        link(first[image_id], first[evidence['representative_id']],
             {key: evidence[key] for key in ('method', 'metrics')})

    # Suppress a second candidate only when it aligns directly to a selected photo.
    # Give the individually inspected version first choice of representative.
    selected = set()
    for status in CANDIDATE_PRIORITY:
        for image_id, decision in decisions.items():
            if decision['status'] != status:
                continue
            match_id = next((other for other in similar[image_id] if other in selected), '')
            if match_id:
                decision.update(status='near_duplicate_aligned', aligned_match_id=match_id,
                                alignment_evidence=similar[image_id][match_id],
                                note='Direct aligned near-copy of a candidate retained for review.')
            else:
                selected.add(image_id)

    for image_id, decision in decisions.items():
        if decision['status'] != 'hold_automatic_uncertain':
            continue
        match_id = next((other for other in similar[image_id] if other in selected), '')
        if match_id:
            decision.update(status='near_duplicate_aligned', aligned_match_id=match_id,
                            alignment_evidence=similar[image_id][match_id],
                            note='Direct aligned near-copy of a candidate retained for review.')

    exported = []
    for row in rows:
        image_id = row['id']
        group_id = first[image_id]
        decision = decisions[group_id]
        representative = decision['image_id']
        canonical = image_id == representative
        status = decision['status'] if canonical else 'exact_duplicate'
        hint = CLASSES[row['expected_class_id']]
        output = ''
        if status == 'reviewed_keep':
            output = f"images/{hint['name_en']}/{image_id}.jpg"
        aligned_id = decision['aligned_match_id'] if canonical else ''
        exported.append(dict(
            id=image_id, filename=row['filename'], source_folder=row['source_folder'],
            source_class_id=hint['class_id'], source_hint_th=hint['name_th'], source_hint_en=hint['name_en'],
            decision=status, decision_th=STATUS_TH[status], group_decision=decision['status'],
            note=decision['note'] if canonical else 'Exact decoded-pixel copy of the representative.',
            visually_reviewed=canonical and decision['visually_reviewed'],
            visual_decision=decision['visual_decision'] if canonical else '',
            representative_id=representative, aligned_match_id=decisions[aligned_id]['image_id'] if aligned_id else '',
            alignment_evidence=json.dumps(decision['alignment_evidence'], separators=(',', ':')) if aligned_id else '',
            source_class_conflict=decision['source_class_conflict'],
            conflict_match_ids=decision['conflict_match_ids'],
            pixel_sha256=row['pixel_sha256'], auto_decision=decision['auto_decision'],
            source_image=f"{row['dataset_root']}/{row['image']}", selected_image=output,
            disease_confirmed=False, label_ready=False,
        ))
    assert len(exported) == len(rows)
    assert sum(row['visually_reviewed'] for row in exported) == len(visual)
    return exported, len(groups)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def publish():
    rows, unique_contents = collect_decisions()
    picked = [row for row in rows if row['selected_image']]
    counts = Counter(row['decision'] for row in rows)
    assert len({row['pixel_sha256'] for row in picked}) == len(picked)
    assert len({row['selected_image'] for row in picked}) == len(picked)
    assert not any(row['source_class_conflict'] for row in picked)
    assert counts['exact_duplicate'] == len(rows) - unique_contents
    reviewed = sum(row['visually_reviewed'] for row in rows)
    report = dict(
        all_files=len(rows), exact_decoded_contents=unique_contents,
        individually_screened=reviewed, automated_only_unique=unique_contents - reviewed,
        selected_images=len(picked), status_counts=counts,
        class_conflict_files=sum(row['source_class_conflict'] for row in rows),
        labels_exported=0, source_images_changed=False,
        by_source_class={
            item['name_en']: dict(Counter(row['decision'] for row in rows
                                        if row['source_class_id'] == item['class_id']))
            for item in CLASSES
        },
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    folder = BASE / NAME
    zip_path = BASE / f'{NAME}.zip'
    if folder.exists() or zip_path.exists():
        raise FileExistsError(f'Existing delivery at {folder} or {zip_path}')
    with tempfile.TemporaryDirectory(dir=BASE) as scratch:
        root = Path(scratch) / NAME
        root.mkdir()
        files = ['README.txt', 'all_files.csv', 'selected_images.csv', 'summary.json']
        for row in picked:
            source, target = BASE / row['source_image'], root / row['selected_image']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            assert sha256(source) == sha256(target), row['id']
            files.append(row['selected_image'])
        write_csv(root / 'all_files.csv', list(rows[0]), rows)
        write_csv(root / 'selected_images.csv', list(rows[0]), picked)
        write_json(root / 'summary.json', report)
        breakdown = '\n'.join(f'{STATUS_TH[status]} {counts[status]:,} ไฟล์' for status in STATUS_TH)
        (root / 'README.txt').write_text(
            'ภาพข้าวที่คัดก่อนทำ label รอบตรวจครบทุกไฟล์\n\n'
            f"คัดครบ {len(rows):,} ไฟล์ใน 9 โรคเดิมและ Tungro (โรคใบสีส้ม)\n"
            f"พิกเซลต่างกัน {unique_contents:,} ภาพ เปิดตรวจคุณภาพรายภาพ {reviewed:,} ภาพ\n"
            f"ภาพที่เหลือ {unique_contents - reviewed:,} ภาพผ่านการคัดอัตโนมัติและแยกผลรายไฟล์\n\n"
            f"{breakdown}\n\n"
            'images รวมเฉพาะภาพที่เปิดดูแล้วและมีรายละเอียดให้เริ่มตรวจชื่อโรคและทำกรอบ\n'
            'auto_candidate ใน all_files.csv คือภาพที่ผ่านเกณฑ์เบื้องต้นและยังพักไว้นอกชุดภาพคัด\n'
            'selected_images.csv เชื่อมไฟล์ที่คัดกับชื่อไฟล์ต้นทางและหมวดโรค\n'
            'all_files.csv บันทึกผลทุกไฟล์ รวมภาพพัก ภาพที่ตัด และรหัสภาพตัวแทน\n'
            'ภาพใกล้ซ้ำจากการจัดแนวอาจอ้างถึงภาพตัวแทนที่พักไว้ ส่วนภาพใกล้ซ้ำจากการเปิดดูบันทึกเหตุผลไว้ใน note\n\n'
            'ชื่อโรคในชื่อโฟลเดอร์และคอลัมน์ source_hint มาจากชุดต้นทางและรอการยืนยัน\n'
            'ชุดนี้สำหรับตรวจชื่อโรคและวาด label ใหม่เฉพาะรอยหรือบริเวณอาการที่เห็นก่อนเทรน\n',
            encoding='utf-8',
        )
        packed = Path(scratch) / zip_path.name
        with zipfile.ZipFile(packed, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for path in files:
                compression = zipfile.ZIP_STORED if path.endswith('.jpg') else zipfile.ZIP_DEFLATED
                archive.write(root / path, f'{NAME}/{path}', compress_type=compression)
        with zipfile.ZipFile(packed) as archive:
            assert set(archive.namelist()) == {f'{NAME}/{path}' for path in files}
            assert archive.testzip() is None
        root.rename(folder)
        packed.rename(zip_path)
    print(f'{zip_path} ({zip_path.stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    publish()
