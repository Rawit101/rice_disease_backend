# -*- coding: utf-8 -*-
"""Resumable large-dataset inference. Predictions are never expert ground truth.

Source images are read only. Normalized copies are the coordinate reference.
Unsupported classes and healthy-source images never become negative labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from PIL import Image, ImageOps

from auto_label_rice import (EXTENSIONS, FOLDERS, NAMES, ORIGINAL_NAMES,
    check_model_names, digest_file, polygons_from_result, write_json, yolo_text)
from build_full_rice_comparison import merge_predictions
from prepare_rice_training import (agreement, copy_file, leakage_groups,
    phash_variants, read_polygons, yaml_text)


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def log(**items):
    print(json.dumps(items, ensure_ascii=False), flush=True)


def link(source, destination):
    if not destination.exists():
        copy_file(source, destination)


def prepare(args):
    out = args.output
    for sub in ['images/pool', 'thumbnails', 'cache', 'metadata', 'predictions', 'labels/pool']:
        (out/sub).mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.source.rglob('*') if p.is_file()
                   and not p.name.startswith('.') and p.suffix.lower() in EXTENSIONS)
    log(stage='hashing', source_files=len(files))
    inventory_path = out/'inventory.json'
    inventory = read_json(inventory_path)
    if inventory is None:
        def inventory_item(p):
            folder = p.relative_to(args.source).parts[0]
            name = FOLDERS.get(folder)
            source = str(p.resolve())
            return dict(id=hashlib.sha256(source.encode()).hexdigest()[:20], source=source,
                        source_folder=folder, source_class_id=NAMES.index(name) if name else None,
                        sha256=digest_file(p), source_bytes=p.stat().st_size)
        inventory = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, r in enumerate(pool.map(inventory_item, files), 1):
                inventory.append(r)
                if i % 2000 == 0:
                    log(stage='hashing', completed=i, total=len(files))
        write_json(inventory_path, inventory)
    elif {str(p.resolve()) for p in files} != {r['source'] for r in inventory}:
        raise ValueError('Source file list changed. Choose a new output directory.')
    by_sha = defaultdict(list)
    for r in inventory:
        by_sha[r['sha256']].append(r)
    previous = read_json(args.previous/'manifest.json', [])
    old = {r['sha256']: r for r in previous}
    old_phash = read_json(args.previous.parent/'phash_cache.json', {})

    def normalize(rs):
        r = rs[0]
        meta_path = out/'metadata'/(r['sha256']+'.json')
        meta = read_json(meta_path)
        if meta is not None:
            return meta
        path = out/'images/pool'/(r['id']+'.jpg')
        try:
            if r['sha256'] in old:
                link(args.previous/old[r['sha256']]['image'], path)
                with Image.open(path) as im:
                    im.load()
                    im = im.convert('RGB')
            else:
                with Image.open(r['source']) as source:
                    source.load()
                    im = ImageOps.exif_transpose(source).convert('RGB')
                im.save(path, quality=95)
            thumb = im.copy()
            thumb.thumbnail((384, 300))
            for duplicate in rs:
                link(path, out/'images/pool'/(duplicate['id']+'.jpg'))
                thumb_path = out/'thumbnails'/(duplicate['id']+'.jpg')
                if not thumb_path.exists():
                    thumb.save(thumb_path, quality=82)
            phash = old_phash.get(r['sha256'])
            if phash is None:
                phash = phash_variants(path)
            meta = dict(width=im.width, height=im.height, canonical_id=r['id'],
                        phash=phash, error=None)
        except Exception as exc:
            meta = dict(canonical_id=r['id'], error=type(exc).__name__+': '+str(exc)[:400])
        write_json(meta_path, meta)
        return meta

    phashes = {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (sha, meta) in enumerate(zip(by_sha, pool.map(normalize, by_sha.values())), 1):
            if not meta['error']:
                phashes[sha] = meta['phash']
            if i % 500 == 0:
                log(stage='normalizing', completed=i, unique_total=len(by_sha),
                    elapsed_seconds=round(time.monotonic()-started))
    write_json(out/'phash_cache.json', phashes)
    records = []
    for r in inventory:
        meta = read_json(out/'metadata'/(r['sha256']+'.json'))
        records.append(dict(r, image='images/pool/'+r['id']+'.jpg' if not meta['error'] else None,
                            thumbnail='thumbnails/'+r['id']+'.jpg' if not meta['error'] else None,
                            **{k: v for k, v in meta.items() if k != 'phash'}))
    write_json(out/'prepared.json', records)
    log(stage='prepared', source_files=len(records), unique_images=len(by_sha),
        read_errors=sum(bool(r['error']) for r in records),
        reused_prior_content=sum(r['sha256'] in old for r in records))


def cache_path(out, sha, model, recovery=False):
    return out/'cache'/(sha+'_'+model+('_1280' if recovery else '_800')+'.json')


def infer(args):
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available in this execution environment')
    out = args.output
    records = read_json(out/'prepared.json')
    unique = {r['sha256']: r for r in records if not r['error']}
    known = {r['sha256']: r for r in records if not r['error']
             and r['source_class_id'] is not None and r['source_class_id'] < 9}
    config = dict(models={k: digest_file(p) for k, p in [('v8', args.model), ('v6', args.secondary)]},
                  names=ORIGINAL_NAMES, imgsz=800, conf=.25, max_det=300,
                  recovery_imgsz=1280, recovery_conf=.10, precision='fp16', rect=False,
                  unsupported_inference='not_run_no_compatible_output_class', schema=1)
    config_path = out/'inference_config.json'
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError('Inference settings/model changed. Use a new output directory.')
    write_json(config_path, config)
    for key, model_path in [('v8', args.model), ('v6', args.secondary)]:
        pending = []
        reused = 0
        for sha, r in known.items():
            dest = cache_path(out, sha, key)
            if dest.exists():
                continue
            # Earlier FP32 baselines have the same checkpoint, source SHA,
            # confidence and image size. Retain their provenance separately.
            if key == 'v8':
                old = out.parent/'local_all/cache'/(sha+'_old.json')
            else:
                old = out.parent/'ensemble'/(sha+'.json')
            if old.exists():
                prior_config = read_json(old.parent.parent/'run_config.json' if key == 'v8'
                                         else old.parent/'config.json', {})
                if prior_config.get('model_sha256') == config['models'][key] and prior_config.get('imgsz') == 800 and prior_config.get('confidence', prior_config.get('conf')) == .25:
                    link(old, dest)
                    reused += 1
                    continue
            pending.append((sha, r))
        log(stage='inference', model=key, pending=len(pending), reused=reused, known_unique=len(known))
        write_json(out/('baseline_reuse_'+key+'.json'), dict(reused_files=reused,
            prior_precision='fp32', prior_rect=True, current_precision='fp16',
            same_checkpoint_and_source=True))
        model = YOLO(str(model_path))
        check_model_names(model)
        if pending:
            model.predict(np.zeros((800, 800, 3), np.uint8), imgsz=800, device=args.device,
                          rect=False, quantize=16, verbose=False)
        started = time.monotonic()
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start+args.batch]
            images = []
            for sha, r in batch:
                with Image.open(out/r['image']) as im:
                    images.append(im.convert('RGB'))
            results = model.predict(images, imgsz=800, conf=.25, device=args.device, rect=False, quantize=16,
                                    max_det=300, verbose=False)
            for (sha, r), result in zip(batch, results):
                write_json(cache_path(out, sha, key), polygons_from_result(result, True))
            done = start+len(batch)
            if done % 100 < args.batch or done == len(pending):
                log(stage='inference', model=key, completed=done, total=len(pending),
                    elapsed_seconds=round(time.monotonic()-started),
                    seconds_per_image=round((time.monotonic()-started)/done, 3))
        del model
        if args.device == 'mps':
            torch.mps.empty_cache()
    missing = [s for s in known if not read_json(cache_path(out, s, 'v8'))
               and not read_json(cache_path(out, s, 'v6'))]
    write_json(out/'recovery_selection.json', missing)
    log(stage='recovery_selection', known_unique_without_standard_predictions=len(missing))
    for key, model_path in [('v8', args.model), ('v6', args.secondary)]:
        pending = [s for s in missing if not cache_path(out, s, key, True).exists()]
        if not pending:
            continue
        model = YOLO(str(model_path))
        check_model_names(model)
        started = time.monotonic()
        for i, sha in enumerate(pending, 1):
            with Image.open(out/known[sha]['image']) as im:
                result = model.predict(im.convert('RGB'), imgsz=1280, conf=.10,
                    device=args.device, max_det=300, rect=False, quantize=16, verbose=False)[0]
            write_json(cache_path(out, sha, key, True), polygons_from_result(result, True))
            if i % 50 == 0 or i == len(pending):
                log(stage='recovery', model=key, completed=i, total=len(pending),
                    elapsed_seconds=round(time.monotonic()-started))
        del model
        if args.device == 'mps':
            torch.mps.empty_cache()
    log(stage='inference_complete', known_unique=len(known), recovery_unique=len(missing))


def choose_labels(expected, raw, source_conflict=False):
    """Preserve actual predicted IDs. Keep missing/new/healthy out of training."""
    reasons = []
    if expected is None or expected >= 9:
        quality = ('source_folder_background_unreviewed' if expected == NAMES.index('Healthy')
                   else 'unsupported_new_disease_no_semantic_mask')
        reasons.append('healthy_source_image_not_visually_accepted' if expected == NAMES.index('Healthy')
                       else 'nine_class_models_cannot_label_this_new_disease')
        return [], quality, reasons, False, False
    standard = raw['v8_800'] or raw['v6_800']
    keys = ['v8_800', 'v6_800'] if standard else ['v8_1280', 'v6_1280']
    objects, conflicts = merge_predictions([(k, raw[k]) for k in keys])
    if source_conflict:
        reasons.append('source_folders_disagree_on_class')
    if not objects:
        return [], 'no_detection_not_a_negative_label', reasons+['requires_actual_annotation_no_mask_available'], False, False
    if not standard:
        reasons.append('low_confidence_high_resolution_recovery')
    if any(o['class_id'] != expected for o in objects):
        reasons.append('predicted_class_differs_from_source_folder')
    if conflicts:
        reasons.append('models_disagree_on_overlapping_object_class')
    # Default pool includes all source-consistent standard predictions. A
    # smaller two-model-agreement pool is also provided; neither is ground truth.
    eligible = standard and not reasons
    strict, _ = agreement(raw['v8_800'], raw['v6_800'], expected)
    return objects, 'automatic_pseudo_label_unreviewed', reasons, bool(eligible), bool(eligible and strict)


def group_datasets(args):
    out = args.output
    records = read_json(out/'prepared.json')
    previous = read_json(args.previous/'manifest.json')
    valid = [r for r in records if not r['error']]
    # Joint grouping screens the new pool against the prior reviewed validation
    # set and the prior manually rejected source images, including near variants.
    group_records = [dict(r, folder_class_id=r['source_class_id']) for r in valid]
    source_shas = {r['sha256'] for r in group_records + previous}
    saved = read_json(out/'split_groups.json', {})
    if saved.get('method') == 'phash_flip_rotate_hamming4_v1' and set(saved.get('groups',{})) == source_shas:
        return saved['groups']
    old_phash = read_json(args.previous.parent/'phash_cache.json', {})
    phashes = read_json(out/'phash_cache.json')
    phashes.update(old_phash)
    for r in previous:
        group_records.append(dict(r, image=str((args.previous/r['image']).resolve())))
    write_json(out/'joint_phash_cache.json', phashes)
    log(stage='grouping', source_files=len(records), with_previous=len(group_records))
    groups = leakage_groups(group_records, out, out/'joint_phash_cache.json')
    write_json(out/'split_groups.json', dict(method='phash_flip_rotate_hamming4_v1', groups=groups))
    log(stage='grouping_complete',unique_sources=len(groups),groups=len(set(groups.values())))
    return groups


def export(args):
    out = args.output
    primary_sha = digest_file(args.model)
    secondary_sha = digest_file(args.secondary)
    inference_config = read_json(out/'inference_config.json')
    if inference_config and inference_config['models'] != {'v8':primary_sha,'v6':secondary_sha}:
        raise ValueError('A checkpoint changed during inference. Do not mix predictions and weights.')
    records = read_json(out/'prepared.json')
    previous = read_json(args.previous/'manifest.json')
    valid = [r for r in records if not r['error']]
    groups = group_datasets(args)
    legacy = {}
    for key in ['v8','v6']:
        reused = set()
        for sha in {r['sha256'] for r in valid}:
            baseline = (out.parent/'local_all/cache'/(sha+'_old.json') if key=='v8'
                        else out.parent/'ensemble'/(sha+'.json'))
            current = cache_path(out,sha,key)
            if baseline.exists() and current.exists() and os.path.samefile(baseline,current):
                reused.add(sha)
        legacy[key] = reused
        write_json(out/('baseline_reuse_'+key+'.json'), dict(reused_files=len(reused),
            source_sha256=sorted(reused),prior_precision='fp32',prior_rect=True,
            current_precision='fp16',same_checkpoint_and_source=True))
    reviewed = {}
    for r in previous:
        if r['quality'] in {'ai_visually_screened_mask','ai_visually_screened_background'}:
            reviewed[r['sha256']] = dict(
                objects=(read_polygons(args.previous/r['label'])
                         if r['quality']=='ai_visually_screened_mask' else []),
                quality=r['quality'], expected=r['source_class_id'],
                prior_id=r['id'], source='prior_visual_review')
    additions = read_json(args.previous.parent/'remaining_recheck/manual_accepted.json', [])
    for r in additions:
        if r.get('visually_accepted'):
            reviewed[r['source_sha256']] = dict(objects=r['objects'],
                quality='ai_visually_screened_mask', expected=r['objects'][0]['class_id'],
                prior_id=r['id'], source='prior_remaining_image_manual_review')
    val_groups = {groups[r['sha256']] for r in previous if r['default_split'] == 'val'}
    bad_groups = {groups[r['sha256']] for r in previous
                  if 'visual_review_previously_found_mask_problems' in r['reasons']}
    current_review = read_json(out/'visual_review_decisions.json', [])
    bad_groups.update(groups[r['sha256']] for r in current_review
                      if r.get('status') == 'reject' and r['sha256'] in groups)
    class_groups = defaultdict(set)
    by_sha = defaultdict(list)
    for r in valid:
        by_sha[r['sha256']].append(r)
        class_groups[groups[r['sha256']]].add(r['source_class_id'])
    conflict_groups = {g for g, classes in class_groups.items() if len(classes) > 1}
    train = []; strict = []; all_predictions = []; manifest = []
    for i, (sha, rs) in enumerate(by_sha.items(), 1):
        rs.sort(key=lambda x: (x['source_class_id'] is None or x['source_class_id'] >= 9, x['source']))
        r = rs[0]
        known = any(x['source_class_id'] is not None and x['source_class_id'] < 9 for x in rs)
        raw = {key: read_json(cache_path(out, sha, key[:2], key.endswith('1280')), [])
               for key in ['v8_800','v6_800','v8_1280','v6_1280']}
        if known and any(not cache_path(out, sha, k).exists() for k in ['v8','v6']):
            raise ValueError('Inference incomplete for '+sha)
        if known and not (raw['v8_800'] or raw['v6_800']) and any(
                not cache_path(out, sha, k, True).exists() for k in ['v8','v6']):
            raise ValueError('Recovery incomplete for '+sha)
        objects, quality, reasons, eligible, consensus = choose_labels(r['source_class_id'], raw,
            len({x['source_class_id'] for x in rs}) > 1)
        prior_review = reviewed.get(sha)
        if (prior_review and len({x['source_class_id'] for x in rs}) == 1
                and prior_review['expected'] == r['source_class_id']):
            objects = prior_review['objects']
            quality = prior_review['quality']
            reasons = []
            eligible = True
            consensus = False  # These labels came from review, not agreement.
        group = groups[sha]
        if group in conflict_groups:
            reasons.append('near_duplicate_group_has_conflicting_source_classes')
            eligible = consensus = False
        if group in bad_groups:
            reasons.append('visual_review_previously_found_mask_problems')
            eligible = consensus = False
        if group in val_groups:
            reasons.append('related_image_in_checked_validation')
            eligible = consensus = False
        for j, r in enumerate(rs):
            image_id = r['id']
            own_objects, own_quality, own_reasons = objects, quality, list(reasons)
            if r['source_class_id'] != rs[0]['source_class_id']:
                own_objects, own_quality, individual_reasons, _, _ = choose_labels(r['source_class_id'], raw,
                    len({x['source_class_id'] for x in rs}) > 1)
                holds = {'near_duplicate_group_has_conflicting_source_classes',
                    'visual_review_previously_found_mask_problems','related_image_in_checked_validation'}
                own_reasons = list(dict.fromkeys(individual_reasons + [x for x in reasons if x in holds]))
            label = ('labels/pool/'+image_id+'.txt' if own_objects
                     or own_quality == 'ai_visually_screened_background' else None)
            if label:
                (out/label).write_text(yolo_text(own_objects), encoding='utf-8')
            path = './images/pool/'+image_id+'.jpg'
            split = 'train' if j == 0 and eligible else None
            if split:
                train.append(path)
            if j == 0 and consensus:
                strict.append(path)
            if j == 0 and label and group not in val_groups:
                all_predictions.append(path)
            prediction = 'predictions/'+image_id+'.json'
            write_json(out/prediction, dict(objects=own_objects, raw_models=raw,
                recovery_inference_run=known and cache_path(out,sha,'v8',True).exists(),
                baseline_precision={k:('fp32_reused' if sha in legacy[k] else 'fp16')
                    for k in ['v8','v6']} if known else None,
                inference_skipped=not known,
                all_masks_unreviewed=not own_quality.startswith('ai_visually_screened'),
                previous_review=prior_review if own_quality.startswith('ai_visually_screened') else None))
            manifest.append(dict(r, label=label, prediction_file=prediction,
                canonical_id=rs[0]['id'],
                source_relative_path=Path(r['source']).relative_to(args.source).as_posix(),
                quality=own_quality, reasons=own_reasons, default_split=split,
                exact_duplicate=j > 0, split_group=group, object_count=len(own_objects),
                class_ids=sorted({o['class_id'] for o in own_objects}),
                two_model_agreement_candidate=bool(j == 0 and consensus),
                expert_diagnosis_confirmed=False))
        if i % 1000 == 0:
            log(stage='exporting', completed=i, unique_total=len(by_sha))
    for r in records:
        if r['error']:
            manifest.append(dict(r, label=None, prediction_file=None, quality='read_error',
                reasons=[r['error']], default_split=None, exact_duplicate=False,
                object_count=0, class_ids=[], expert_diagnosis_confirmed=False))
    val = []
    val_manifest = []
    for r in previous:
        if r['default_split'] != 'val':
            continue
        link(args.previous/r['image'], out/r['image'])
        link(args.previous/r['label'], out/r['label'])
        val.append('./'+r['image'])
        val_manifest.append(dict(r, split_group=groups[r['sha256']]))
    write_json(out/'validation_manifest.json',val_manifest)
    for name, paths in [('train',train),('strict_train',strict),('all_prediction_train',all_predictions),('val',val)]:
        (out/(name+'.txt')).write_text('\n'.join(paths)+'\n', encoding='utf-8')
    for name, train_list in [('data.yaml','train.txt'),('data_strict.yaml','strict_train.txt'),
                             ('data_all_predictions.yaml','all_prediction_train.txt')]:
        (out/name).write_text(yaml_text(train_list,'val.txt'), encoding='utf-8')
    (out/'weights').mkdir(exist_ok=True)
    checkpoint_copy = out/'weights/best.pt'
    if not checkpoint_copy.exists():
        shutil.copy2(args.model, checkpoint_copy)
    elif digest_file(checkpoint_copy) != primary_sha:
        raise ValueError('Existing exported checkpoint differs from inference checkpoint')
    fields = ['id','source_folder','source_class_id','quality','default_split','object_count',
              'exact_duplicate','source','image','label','reasons']
    with (out/'image_status.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in manifest:
            writer.writerow({k: '; '.join(r[k]) if k == 'reasons' else r.get(k) for k in fields})
    report = dict(total_source_images_compared=len(records),unique_source_contents=len(by_sha),
        source_images_accounted_for=len(records),
        files_in_supported_categories=sum(r['source_class_id'] is not None and r['source_class_id']<9 for r in records),
        teacher_inference_skipped_for_unsupported_categories=True,
        source_files_with_model_or_checked_annotations=sum(bool(r['label']) for r in manifest),
        source_files_without_semantic_annotations=sum(not r['label'] for r in manifest),
        status_counts=dict(Counter(r['quality'] for r in manifest)),
        folder_counts=dict(Counter(r['source_folder'] for r in manifest)),
        class_names=ORIGINAL_NAMES, default_train_images=len(train),
        two_model_agreement_train_images=len(strict), raw_prediction_train_images=len(all_predictions),
        prior_checked_validation_images=len(val),all_newly_inferred_labels_unreviewed=True,
        previously_reviewed_label_files_reused=sum(r['quality'].startswith('ai_visually_screened') for r in manifest),
        primary_model_sha256=primary_sha,
        secondary_model_sha256=secondary_sha,
        new_class_masks_created=0, unreviewed_healthy_negative_labels_created=0,
        training_executed=False,validation_covers_only_previous_screened_subset=True)
    write_json(out/'manifest.json',manifest)
    write_json(out/'report.json',report)
    per_folder=[]
    for folder in sorted(report['folder_counts']):
        rows=[r for r in manifest if r['source_folder']==folder]
        per_folder.append(dict(source_folder=folder,source_files=len(rows),
            label_files=sum(bool(r['label']) for r in rows),
            positive_mask_files=sum(r['object_count']>0 for r in rows),
            train_unique=sum(r['default_split']=='train' for r in rows),
            without_label=sum(not r['label'] for r in rows)))
    with (out/'folder_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(per_folder[0]));writer.writeheader();writer.writerows(per_folder)
    summary=[
        '# ผลจัดชุดเพิ่มเติม Rice Disease', '',
        f"- รูปต้นฉบับ {len(records):,} ไฟล์ / เนื้อหาไม่ซ้ำ {len(by_sha):,} ชุด",
        f"- มีไฟล์ label {report['source_files_with_model_or_checked_annotations']:,} ไฟล์ (รวมภาพปกติที่เคยตรวจไว้แล้ว)",
        f"- รายการ train หลัก {len(train):,} ภาพไม่ซ้ำ / กลุ่มสองโมเดลเห็นตรงกัน {len(strict):,} ภาพ",
        f"- ชุด val เดิมที่ AI เคยตรวจ {len(val)} ภาพ ครอบคลุมเพียงบางคลาส ไม่ใช่ชุดทดสอบอิสระครบทุกโรค",
        f"- ยังไม่มี label {report['source_files_without_semantic_annotations']:,} ไฟล์ ดูเหตุผลรายภาพใน image_status.csv", '',
        'label ใหม่เป็นผลอัตโนมัติ ยังไม่ใช่คำยืนยันจากผู้เชี่ยวชาญ ต้องตรวจคุณภาพก่อนใช้เป็น ground truth',
        'ใช้ data.yaml หรือ data_strict.yaml และรายการภาพที่ให้มา ไม่ชี้ train ไปทั้ง images/pool',
        'โรคใหม่ไม่มีขอบเขตโรค เพราะโมเดลเดิมมีเพียง 9 คลาส ภาพ Healthy ที่ยังไม่ตรวจไม่ถูกสร้างเป็น negative label',
        'เก็บทุกรูปไว้ครบ รวมผลที่ชื่อโรคขัดกัน แต่กันออกจากรายการ train หลักตามเหตุผลที่บันทึกไว้', '',
        '| หมวดต้นฉบับ | รูป | มีไฟล์ label | เข้า train หลัก | ยังไม่มี label |',
        '|---|---:|---:|---:|---:|']
    for row in per_folder:
        summary.append('| {source_folder} | {source_files:,} | {label_files:,} | {train_unique:,} | {without_label:,} |'.format(**row))
    (out/'SUMMARY_TH.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    shutil.copyfile(Path(__file__).with_name('rice_archive_compare.html'),out/'compare.html')
    write_readme(out,report)
    for name in ['validate_rice_archive.py','merge_rice_training.py','restore_rice_images.py']:
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    log(stage='export_complete',**report)


def write_readme(out, report):
    text = '''# Rice Disease: automatic segmentation candidates

All input files are accounted for in manifest.json and image_status.csv.
The original Downloads images were not modified. Use images/pool copies with
the supplied polygons: EXIF orientation was applied before coordinates were made.

New model masks are unreviewed predictions, not expert-confirmed ground truth.
Prior AI-reviewed labels are reused for exact source SHA matches when the source
category agrees, including the earlier single-image manual boundary correction.
data.yaml uses unique source-consistent standard-resolution model predictions
and available prior AI-reviewed labels. Expert confirmation is false for both.
data_strict.yaml uses a smaller pool where both models agree on class, score,
instance count and boundary. Agreement does not establish correctness.
data_all_predictions.yaml includes conflicts and weak recovery results for
repair/review. Do not assume the broadest list is the best training set.

Missing detection, unsupported diseases and unreviewed Healthy images have
NO training label; they are never silently treated as negative images.
Use the supplied YAML/text lists. Do NOT set train=images/pool: generic YOLO
directory scanning would interpret missing annotations as negative examples.
The 18 source folders are image-level category hints. The two models can output
only the preserved 9 old classes (class IDs 0-8). Neck Blast is kept separate
from Leaf Blast. New diseases require semantic annotation and expanded heads.

val.txt contains the prior AI-screened 34-image subset. It is tiny and covers
only some old classes. It does NOT measure performance across all 18 categories.
Exact SHA and conservative flip/rotation pHash groups screen the new training
pool against this validation subset. Collection independence is not established.
Prefer the original dataset's independent validation/test sets when merging.

Transfer this whole directory, including images, labels, text lists and weights.
Install the friend's compatible Ultralytics environment. A starting command:
  yolo segment train model=weights/best.pt data=data.yaml imgsz=800 epochs=50
Run from this directory. This project has NOT started training or changed best.pt.

View: python -m http.server 8770 --bind 127.0.0.1
Then open http://127.0.0.1:8770/compare.html

Outputs: images/pool (all decodable sources + prior validation), labels/pool,
predictions, train.txt, strict_train.txt, all_prediction_train.txt, val.txt,
data*.yaml, class_mapping.json, image_status.csv, manifest.json, report.json.
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    write_json(out/'class_mapping.json', dict(old_model_names=ORIGINAL_NAMES,
        source_folder_to_name=FOLDERS, provisional_new_names=NAMES[9:],
        additional_segmentation_classes_not_trained=True))


def package(args):
    out = args.output
    target = out.with_suffix('.zip')
    allowed = ['images','labels','predictions','thumbnails','weights']
    with zipfile.ZipFile(target,'w',allowZip64=True) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name not in {'prepared.json','inventory.json','phash_cache.json','joint_phash_cache.json','progress.html'} and p.suffix in {'.json','.csv','.yaml','.html','.md','.txt','.py'}:
                z.write(p,p.name,compress_type=zipfile.ZIP_DEFLATED)
        for folder in allowed:
            for p in sorted((out/folder).rglob('*')):
                if p.is_file() and p.suffix!='.cache':
                    z.write(p,str(p.relative_to(out)),compress_type=(zipfile.ZIP_STORED
                            if p.suffix in {'.jpg','.pt'} else zipfile.ZIP_DEFLATED))
    log(stage='packaged',path=str(target),bytes=target.stat().st_size)
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad:
            raise ValueError('ZIP CRC failed: '+bad)
    log(stage='zip_crc_passed')
    slim = out.parent/(out.name+'_labels_only.zip')
    with zipfile.ZipFile(slim,'w',allowZip64=True) as z:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name not in {'prepared.json','inventory.json','phash_cache.json','joint_phash_cache.json','progress.html'} and p.suffix in {'.json','.csv','.yaml','.html','.md','.txt','.py'}:
                z.write(p,p.name,compress_type=zipfile.ZIP_DEFLATED)
        for folder in ['labels','predictions','weights']:
            for p in sorted((out/folder).rglob('*')):
                if p.is_file() and p.suffix!='.cache':
                    z.write(p,str(p.relative_to(out)),compress_type=(zipfile.ZIP_STORED
                            if p.suffix=='.pt' else zipfile.ZIP_DEFLATED))
        for r in read_json(out/'validation_manifest.json'):
            z.write(out/r['image'],r['image'],compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(slim) as z:
        bad=z.testzip()
        if bad:
            raise ValueError('Small ZIP CRC failed: '+bad)
    log(stage='labels_only_zip_crc_passed',path=str(slim),bytes=slim.stat().st_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','infer','group','export','validate','package','all','finish'])
    p.add_argument('--source',type=Path,default=Path('/Users/ssrawit/Downloads/Rice Disease'))
    p.add_argument('--output',type=Path,default=Path('temp/rice_labeling/rice_archive_auto'))
    p.add_argument('--previous',type=Path,default=Path('temp/rice_labeling/rice_full_auto'))
    p.add_argument('--model',type=Path,default=Path('best.pt'))
    p.add_argument('--secondary',type=Path,default=Path('/Users/ssrawit/Downloads/rice9disease_v6_BG_Polygon/weights/best.pt'))
    p.add_argument('--device',default='mps',choices=['cpu','mps'])
    p.add_argument('--batch',type=int,default=8)
    args = p.parse_args()
    for key in ['source','output','previous','model','secondary']:
        setattr(args,key,getattr(args,key).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action == 'finish':
        deadline=time.monotonic()+1800
        while not (args.output/'prepared.json').exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Preparation did not complete within 30 minutes')
            time.sleep(5)
        # Each stage imports the current code afresh, so review/export fixes do
        # not require interrupting a long checkpointed GPU inference stage.
        for stage in ['infer','export','validate','package']:
            command=[sys.executable,str(Path(__file__).resolve()),stage]
            for key in ['source','output','previous','model','secondary','device','batch']:
                command.extend(['--'+key,str(getattr(args,key))])
            subprocess.run(command,check=True)
        return
    from validate_rice_archive import validate,validate_loader
    def validate_export(args):
        log(stage='validating_dataset')
        validate(args.output)
        validate_loader(args.output)
    for action, func in [('prepare',prepare),('infer',infer),('group',group_datasets),('export',export),
                         ('validate',validate_export),('package',package)]:
        if args.action in [action,'all'] or (args.action=='finish' and action!='prepare'):
            func(args)


if __name__ == '__main__':
    main()
