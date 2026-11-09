"""Build a portable YOLO segmentation training set from visually reviewed labels."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import SegmentationModel

from publish_rice_class_names import BASE, CLASSES, NAMES, parse_label, write_csv
from publish_rice_relabel_review import simple_polygon


SOURCE = BASE / 'rice_relabel_review'
WORK = BASE / 'rice_training_work'
OUT = BASE / 'rice_yolo10_finetune'
WEIGHTS = BASE.parents[1] / 'best.pt'
SOURCE_REVIEW = BASE / 'rice_relabel_work' / 'inventory.json'
HELD_TUNGRO = {
    '0d8fdc43bd22bd064557': 'Off-leaf islands and missed discolored tip.',
    '8c10137e329ec41d9214': 'The outline includes a large green blade portion.',
    '600ceed6a7ea581dbe19': 'One outline covers mainly a green blade edge.',
    'f658bcf42a73755a4200': 'An upper-right contour spills onto gray soil.',
    '22c7b805bcbf476927d2': 'The left contour spills around the blade.',
    '6fa192a76a5a104e1a46': 'The outline extends over the dirt beside the blade.',
    'e785e62828716bb24fdc': 'An isolated contour includes green leaf or soil.',
}
OTHER_HOLDS = {
    '16f8cb4c0bfbb46caa6d': 'The annotated brown spot is too faint to resolve clearly.',
    'd5bb0b4617eaa8a39d16': 'The white streak has no distinct dark lesion margin.',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def source_review_table(root, decisions):
    by_id = {row['id']: row for row in decisions}
    with (SOURCE / 'all_review_results.csv').open(encoding='utf-8-sig', newline='') as stream:
        original = list(csv.DictReader(stream))
    for row in original:
        latest = by_id.get(row['id'])
        row.update(
            latest_training_decision=latest['decision'] if latest else 'held_from_previous_review',
            training_split=latest['split'] if latest else '',
            latest_loader_ready=bool(latest and latest['split']),
            latest_training_reason=latest['review_note'] if latest else row['visual_note'],
        )
    write_csv(root / 'all_source_review.csv', list(original[0]), original)


def review_candidates():
    inventory = read_json(SOURCE_REVIEW)
    group_report = read_json(WORK / 'split_audit.json')
    revisions = read_json(WORK / 'tungro_revisions.json')
    assert revisions.keys() <= HELD_TUNGRO.keys()
    reviewed, labels = [], {}
    with (SOURCE / 'draft_index.csv').open(encoding='utf-8-sig', newline='') as stream:
        for source in csv.DictReader(stream):
            image_id = source['id']
            cls = int(source['class_hint_id'])
            photo = inventory[image_id]
            revision = revisions.get(image_id)
            status = 'use_reviewed_label'
            note = source['visual_note']
            if revision:
                status = 'use_redrawn_label' if revision['status'] == 'redo' else 'hold'
                note = revision['note']
            elif image_id in OTHER_HOLDS or image_id in HELD_TUNGRO:
                status = 'hold'
                note = {**OTHER_HOLDS, **HELD_TUNGRO}[image_id]
            elif cls == 9 and source['filename'].startswith('IMG_'):
                status = 'hold'
                note = 'Isolated blade on paper, with dry or linear marks and insufficient Tungro context.'
            split_group = group_report['groups'][image_id]
            row = dict(
                id=image_id, filename=source['filename'], class_id=cls,
                class_name_th=CLASSES[cls]['name_th'], class_name_en=NAMES[cls],
                source_image=source['source_image'], input_annotation=source['packaged_label'],
                original_review_status=source['status'], decision=status, review_note=note,
                split_group=split_group, pixel_sha256=photo['pixel_sha256'],
                disease_identity_expert_confirmed=False, split='', image='', label='', object_count=0,
            )
            if status != 'hold':
                if revision:
                    label = '\n'.join(
                        f"{cls} " + ' '.join(f'{coord:.7f}' for point in p for coord in point)
                        for p in revision['polygons']
                    ) + '\n'
                else:
                    label = (SOURCE / source['packaged_label']).read_text(encoding='utf-8')
                objects = parse_label(label, NAMES)
                assert objects and all(o['class_id'] == cls and simple_polygon(o['polygon'])
                                       for o in objects), image_id
                row['object_count'] = len(objects)
                labels[image_id] = label
            reviewed.append(row)
    assert len(reviewed) == 140
    return reviewed, labels, group_report


def assign_splits(rows, validation_group_allowlist):
    grouped = defaultdict(list)
    for row in rows:
        if row['decision'] != 'hold':
            grouped[row['split_group']].append(row)
    counts = Counter(r['class_id'] for group in grouped.values() for r in group)
    assert set(counts) == set(range(len(NAMES))), counts
    assert len({r['pixel_sha256'] for group in grouped.values() for r in group}) == sum(counts.values())
    validation = set()
    for cls, total in sorted(counts.items()):
        keys = [key for key, group in grouped.items() if {r['class_id'] for r in group} == {cls}]
        if cls == 9 and validation_group_allowlist is not None:
            keys = [key for key in keys if key in validation_group_allowlist]
        # Prefer the closest group-wise count to 20%, keeping at least one train image.
        totals = {0: ()}
        ordered = sorted(keys, key=lambda key: hashlib.sha256(('rice10:' + key).encode()).hexdigest())
        for key in ordered:
            for n, used in list(totals.items()):
                next_n = n + len(grouped[key])
                if next_n < total:
                    totals.setdefault(next_n, used + (key,))
        candidates = [n for n in totals if n]
        if candidates:
            goal = max(1, round(total * .2))
            best = min(candidates, key=lambda n: (abs(n - goal), n > goal))
            validation.update(totals[best])
    for group_id, group in grouped.items():
        split = 'val' if group_id in validation else 'train'
        for row in group:
            image_id = row['id']
            row.update(split=split, image=f'images/{split}/{image_id}.jpg',
                       label=f'labels/{split}/{image_id}.txt')
    assert {r['class_id'] for r in rows if r['split'] == 'train'} == set(range(len(NAMES)))
    assert not ({r['split_group'] for r in rows if r['split'] == 'train'} &
                {r['split_group'] for r in rows if r['split'] == 'val'})


def validate_loader(root, rows):
    data = check_det_dataset(str((root / 'data.yaml').resolve()), autodownload=False)
    assert list(data['names'].values()) == NAMES and data['nc'] == len(NAMES)
    results = {}
    for split in ('train', 'val'):
        dataset = YOLODataset(
            img_path=data[split], data=data, task='segment', imgsz=800, augment=False,
            hyp=DEFAULT_CFG, cache=False, rect=False, batch_size=4, prefix=f'{split}: ',
        )
        expected = {r['id']: r for r in rows if r['split'] == split}
        assert len(dataset) == len(expected) and expected
        for index, record in enumerate(dataset.labels):
            row = expected[Path(record['im_file']).stem]
            assert len(record['segments']) == len(record['cls']) == row['object_count']
            assert set(record['cls'].ravel().tolist()) == {row['class_id']}
            item = dataset[index]
            assert len(item['cls']) == row['object_count'] and torch.count_nonzero(item['masks'])
        results[split] = {
            'images': len(dataset), 'polygons': sum(r['object_count'] for r in expected.values()),
            'class_ids': sorted({r['class_id'] for r in expected.values()}),
        }
        (root / 'labels' / f'{split}.cache').unlink(missing_ok=True)
    return results


def validate_checkpoint(root):
    source = YOLO(str(root / 'weights' / 'best.pt'))
    assert source.task == 'segment' and list(source.names.values()) == NAMES[:9]
    target = SegmentationModel(deepcopy(source.model.yaml), nc=len(NAMES), verbose=False)
    target.names = dict(enumerate(NAMES))
    target.load(source.model)
    old_head, new_head = source.model.model[-1], target.model[-1]
    assert old_head.nc == 9 and new_head.nc == len(NAMES)
    matched = 0
    for branch in ('cv3', 'one2one_cv3'):
        for before, after in zip(getattr(old_head, branch), getattr(new_head, branch)):
            for parameter in ('weight', 'bias'):
                assert torch.equal(getattr(before[-1], parameter), getattr(after[-1], parameter)[:9])
                matched += 1
    return dict(task=source.task, pretrained_class_count=9, target_class_count=10,
                matching_class_head_tensors=matched, ultralytics=ultralytics_version)


def stage(root):
    torch.set_num_threads(2)
    rows, labels, group_report = review_candidates()
    assign_splits(rows, group_report['candidate_tungro_val_groups'])
    root.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            (root / kind / split).mkdir(parents=True)
    for row in rows:
        if row['split']:
            shutil.copyfile(BASE / row['source_image'], root / row['image'])
            (root / row['label']).write_text(labels[row['id']], encoding='utf-8')
    (root / 'weights').mkdir()
    shutil.copyfile(WEIGHTS, root / 'weights' / 'best.pt')
    (root / 'data.yaml').write_text(yaml.safe_dump({
        'train': 'images/train', 'val': 'images/val',
        'nc': len(NAMES), 'names': dict(enumerate(NAMES)),
    }, sort_keys=False), encoding='utf-8')
    write_csv(root / 'review_decisions.csv', list(rows[0]), rows)
    source_review_table(root, rows)
    write_csv(root / 'classes.csv', list(CLASSES[0]), CLASSES)
    shutil.copyfile(Path(__file__).with_name('rice10_train.py'), root / 'train.py')
    (root / 'requirements.txt').write_text(f'ultralytics=={ultralytics_version}\n', encoding='utf-8')
    by_class = [
        dict(**cls, train=sum(r['split'] == 'train' and r['class_id'] == cls['class_id'] for r in rows),
             val=sum(r['split'] == 'val' and r['class_id'] == cls['class_id'] for r in rows))
        for cls in CLASSES
    ]
    write_csv(root / 'split_counts.csv', list(by_class[0]), by_class)
    loader = validate_loader(root, rows)
    checkpoint = validate_checkpoint(root)
    report = dict(
        technical_training_ready=True, expert_diagnosis_confirmation_pending=True,
        candidate_count=len(rows), decisions=dict(Counter(r['decision'] for r in rows)),
        images_per_class=by_class, loader=loader, checkpoint=checkpoint,
        validation_kind='small provisional group-aware split of additions only',
        old_original_train_and_test_corpus_available=False,
        diagnosis_status='visually screened class candidates, expert confirmation pending',
        class_review_record='review_decisions.csv', group_basis=group_report['base_method'],
    )
    write_json(root / 'validation_report.json', report)
    (root / 'README_TH.txt').write_text(
        'ชุด YOLO Segmentation สำหรับทดลอง fine-tune โรคข้าว 10 คลาส\n\n'
        f"ภาพ train {loader['train']['images']} ภาพ และ val {loader['val']['images']} ภาพ\n"
        'คลาส 0–8 ตรงกับ best.pt เดิม และคลาส 9 คือ Tungro โรคใบสีส้ม\n'
        'ไฟล์ weights/best.pt เป็น checkpoint ต้นทางสำหรับเริ่ม fine-tune\n\n'
        'แตก ZIP แล้วเปิด Terminal ในโฟลเดอร์ rice_yolo10_finetune\n'
        'python -m pip install -r requirements.txt\n'
        'python train.py --epochs 30 --batch 4\n\n'
        'คำสั่งใช้ data.yaml และ best.pt จากโฟลเดอร์ที่แตกไว้ ผลอยู่ใน runs\n'
        'รองรับ --device 0 สำหรับ CUDA, --device mps สำหรับ Apple Silicon หรือ --device cpu\n'
        'split_counts.csv แสดงจำนวน train และ val แยกรายคลาส\n'
        'review_decisions.csv แสดงที่มา เหตุผล และภาพที่พักจากการตรวจซ้ำ\n'
        'all_source_review.csv รวมประวัติการตรวจ 12,564 ไฟล์และผลคัดเข้าชุดเทรนล่าสุด\n\n'
        'ชุดนี้เปิดเทรนได้เพื่อทดลอง คลาสมาจากการดูอาการร่วมกับข้อมูลต้นทางและยังรอผู้รู้ยืนยัน\n'
        'ชุด val ขนาดเล็ก บางโรคมีตัวอย่างน้อย และข้อมูลต้นฉบับจากการเทรน best.pt ยังไม่อยู่ในชุดนี้\n'
        'ใช้ผล val ดูการรันเบื้องต้น การวัดว่าโมเดลเก่งกว่าเดิมต้องมีชุดทดสอบอิสระของโรคเดิมและ Tungro\n'
        'จัดภาพที่ตรวจพบว่าเป็นภาพใกล้ซ้ำหรือมาจากฉากเดียวกันไว้ split เดียวกัน\n',
        encoding='utf-8-sig',
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def pack(root):
    archive = root.with_suffix('.zip')
    temporary = archive.with_suffix('.zip.tmp')
    files = [p for p in sorted(root.rglob('*')) if p.is_file()]
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in files:
            method = zipfile.ZIP_STORED if path.suffix in {'.jpg', '.pt'} else zipfile.ZIP_DEFLATED
            z.write(path, str(path.relative_to(root.parent)), compress_type=method)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {str(p.relative_to(root.parent)) for p in files}
    temporary.replace(archive)
    print(archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('stage', 'pack'))
    parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args()
    (stage if args.command == 'stage' else pack)(args.root.resolve())
