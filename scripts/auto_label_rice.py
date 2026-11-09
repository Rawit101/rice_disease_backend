# -*- coding: utf-8 -*-
"""Generate resumable, explicitly unreviewed rice segmentation drafts.

This never trains or replaces a model. Old-model class IDs are preserved.
Class-agnostic FastSAM masks are saved ONLY as unclassified proposals, never
silently relabeled as diseases or exported as training ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
import zipfile


ORIGINAL_NAMES = [
    "Bacterial_Blight", "Brown_Spot", "Rice_Blast", "Narrow_Brown_Spot",
    "False_Smut", "Dirty_Seed", "Sheath_Rot", "Stem_Rot", "Red_Stripe",
]
NAMES = ORIGINAL_NAMES + [
    "Bakanae", "Bacterial_Streak", "Grassy_Stunt_Virus", "Healthy", "Hispa",
    "Leaf_Scald", "Leaf_Smut", "Neck_Blast", "Ragged_Stunt_Virus",
    "Sheath_Blight", "Tungro", "Root_Knot",
]
FOLDERS = dict(zip([
    "Bacterial Blight", "Brown Spot", "Leaf Blast", "Narrow Brown Spot",
    "False Smut", "Dirty Seed", "Sheath Rot", "Stem Rot", "Red Stripe",
], ORIGINAL_NAMES))
FOLDERS.update({name.replace("_", " "): name for name in NAMES[9:]})
FOLDERS.update(dict(zip([
    "โรคขอบใบเเห้ง", "โรคใบจุดสีน้ำตาล", "โรคไหม้", "โรคใบขีดสีน้ำตาล",
    "โรคดอกกระถิน", "โรคเมล็ดด่าง", "โรคกาบใบเน่า", "โรคลำต้นเน่า", "โรคใบเเถบเเดง",
], ORIGINAL_NAMES)))
FOLDERS.update({"โรคถอดฟักดาบ": "Bakanae", "healthy_rice_plant": "Healthy", "โรครากปม": "Root_Knot"})
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".avif"}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def open_db(output):
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "progress.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS images (
        id TEXT PRIMARY KEY, source TEXT UNIQUE NOT NULL, folder TEXT NOT NULL,
        expected_id INTEGER, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
        record TEXT)""")
    return db


def discover(sources, db):
    added = 0
    for root in sources:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError("Source directory does not exist: " + str(root))
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in EXTENSIONS:
                continue
            source = str(p.resolve())
            if db.execute("SELECT 1 FROM images WHERE source=?", (source,)).fetchone():
                continue
            folder = next((parent.name for parent in p.parents if parent.name in FOLDERS), p.parent.name)
            name = FOLDERS.get(folder)
            expected_id = NAMES.index(name) if name else None
            image_id = hashlib.sha256(source.encode()).hexdigest()[:20]
            db.execute("INSERT INTO images VALUES (?,?,?,?,?,?,NULL)",
                       (image_id, source, folder, expected_id, digest_file(p), p.stat().st_size))
            added += 1
            if added % 100 == 0:
                db.commit()
    db.commit()
    return added


def valid_polygon(points):
    """Validate a normalized segmentation polygon; reject boxes and degenerate masks."""
    if len(points) < 3:
        return None
    values = [[float(x), float(y)] for x, y in points]
    if any(not math.isfinite(v) or v < -0.0001 or v > 1.0001 for pt in values for v in pt):
        return None
    values = [[min(1.0, max(0.0, x)), min(1.0, max(0.0, y))] for x, y in values]
    if len({tuple(x) for x in values}) < 3:
        return None
    area = abs(sum(values[i][0] * values[(i + 1) % len(values)][1] -
                   values[(i + 1) % len(values)][0] * values[i][1]
                   for i in range(len(values)))) / 2
    return values if area > 1e-9 else None


def polygons_from_result(result, semantic):
    objects = []
    if result.masks is None or result.boxes is None:
        return objects
    boxes = result.boxes.cpu()
    for polygon, box in zip(result.masks.xyn, boxes):
        polygon = valid_polygon(polygon)
        if polygon is None:
            continue
        item = {"polygon": polygon, "score": float(box.conf.item())}
        if semantic:
            item["class_id"] = int(box.cls.item())
        objects.append(item)
    return objects


def yolo_text(objects):
    lines = []
    for obj in objects:
        polygon = valid_polygon(obj["polygon"])
        class_id = obj.get("class_id")
        if polygon is None or not isinstance(class_id, int) or not 0 <= class_id < len(NAMES):
            raise ValueError("Invalid semantic polygon or class ID")
        lines.append(str(class_id) + " " + " ".join(format(v, ".8f") for xy in polygon for v in xy))
    return "\n".join(lines) + ("\n" if lines else "")


def classify_record(expected_id, objects, proposals, conflict=False):
    reasons = []
    if expected_id is None:
        reasons.append("unmapped_folder")
    elif expected_id >= len(ORIGINAL_NAMES):
        reasons.append("class_not_supported_by_old_model")
    if conflict:
        reasons.append("identical_content_in_different_classes")
    if objects and expected_id is not None and any(o["class_id"] != expected_id for o in objects):
        reasons.append("model_class_differs_from_folder")
    if objects and min(o["score"] for o in objects) < 0.5:
        reasons.append("low_model_score")
    if not objects:
        reasons.append("no_semantic_label")
    if proposals:
        reasons.append("unclassified_object_masks_are_not_disease_annotations")
    if expected_id == NAMES.index("Healthy"):
        reasons.append("confirm_healthy_class_vs_background_policy")
    status = "semantic_draft" if objects else "unclassified_proposals" if proposals else "needs_annotation"
    return status, reasons


def make_preview(image, objects, proposals, destination):
    from PIL import Image, ImageDraw
    image = image.copy()
    image.thumbnail((720, 540))
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    colors = [(21, 163, 121), (37, 99, 235), (217, 119, 6), (190, 24, 93)]
    for i, obj in enumerate(objects if objects else proposals):
        color = colors[i % len(colors)]
        points = [(x * w, y * h) for x, y in obj["polygon"]]
        draw.polygon(points, fill=(*color, 55), outline=(*color, 255), width=2)
        label = (NAMES[obj["class_id"]] if "class_id" in obj else "UNCLASSIFIED")
        x, y = points[0]
        draw.text((max(0, x), max(0, y - 12)), label, fill=(*color, 255), stroke_width=1, stroke_fill="white")
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.save(destination, quality=85)


def check_model_names(model):
    actual = [model.names[i] for i in range(len(model.names))]
    if actual != ORIGINAL_NAMES or model.task != "segment":
        raise ValueError("The selected model's task/classes differ from the preserved mapping")


def run(args, db):
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    from ultralytics import YOLO, FastSAM

    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = {"model_sha256": digest_file(args.model), "imgsz": args.imgsz,
              "confidence": args.conf, "names": NAMES, "schema": 2}
    config_path = args.output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run settings changed; use a new output directory to avoid mixing drafts")
    write_json(config_path, config)
    if args.proposal_model:
        proposal_config = {"model_sha256": digest_file(args.proposal_model),
                           "imgsz": args.proposal_imgsz, "confidence": 0.4, "max_det": 100}
        proposal_config_path = args.output / "proposal_config.json"
        if proposal_config_path.exists() and json.loads(proposal_config_path.read_text()) != proposal_config:
            raise ValueError("Proposal model/settings changed; use a new output directory")
        write_json(proposal_config_path, proposal_config)
    for name in ["images", "labels_draft", "proposals", "previews", "cache"]:
        (args.output / name).mkdir(exist_ok=True)
    model = YOLO(str(args.model))
    check_model_names(model)
    helper = FastSAM(str(args.proposal_model)) if args.proposal_model else None
    # Warm up outside the real dataset, avoiding initial MPS/NMS compilation timeouts.
    blank = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for _ in range(2):
        model.predict(blank, device=device, imgsz=args.imgsz, conf=args.conf, verbose=False)
    if helper:
        helper.predict(blank, device=device, imgsz=args.proposal_imgsz, verbose=False)
    conflict_hashes = {r[0] for r in db.execute(
        "SELECT sha256 FROM images GROUP BY sha256 HAVING COUNT(DISTINCT expected_id)>1")}
    all_rows = db.execute("SELECT id,source,folder,expected_id,sha256,bytes,record FROM images ORDER BY source").fetchall()
    rows = []
    for row in all_rows:
        previous = json.loads(row[6]) if row[6] else None
        if previous is None or (args.retry_unlabeled and previous["status"] != "error"
                                and not previous["objects"] and not previous["proposal_count"]):
            rows.append(row[:6])
    if args.limit:
        # A pilot should cover different folders, rather than only the first class.
        groups = defaultdict(list)
        for row in rows:
            groups[row[2]].append(row)
        selected = []
        while groups and len(selected) < args.limit:
            for key in list(groups):
                selected.append(groups[key].pop(0))
                if not groups[key]:
                    del groups[key]
                if len(selected) == args.limit:
                    break
        rows = selected
    print(json.dumps({"pending_this_run": len(rows), "device": device}), flush=True)
    started = time.monotonic()
    image_cache = {}
    for index, (image_id, source, folder, expected_id, sha256, size) in enumerate(rows):
        record = {"id": image_id, "source": source, "folder": folder,
                  "folder_class_id": expected_id, "sha256": sha256, "source_bytes": size,
                  "reviewed": False, "image": None, "draft_label": None, "proposal_file": None,
                  "preview": None, "objects": [], "proposal_count": 0}
        try:
            with Image.open(source) as original:
                original.load()
                image = ImageOps.exif_transpose(original).convert("RGB")
            record["width"], record["height"] = image.size
            destination = args.output / "images" / (image_id + ".jpg")
            if sha256 in image_cache:
                if not destination.exists():
                    try:
                        os.link(image_cache[sha256], destination)
                    except OSError:
                        shutil.copyfile(image_cache[sha256], destination)
            elif not destination.exists():
                image.save(destination, quality=95)
                image_cache[sha256] = destination
            record["image"] = "images/" + destination.name
            objects, proposals = [], []
            known = expected_id is not None and expected_id < len(ORIGINAL_NAMES)
            if known:
                cache = args.output / "cache" / (sha256 + "_old.json")
                if cache.exists():
                    objects = json.loads(cache.read_text())
                else:
                    result = model.predict(image, imgsz=args.imgsz, conf=args.conf,
                                           device=device, verbose=False, max_det=300)[0]
                    objects = polygons_from_result(result, semantic=True)
                    write_json(cache, objects)
            if not objects and helper:
                cache = args.output / "cache" / (sha256 + "_objects.json")
                if cache.exists():
                    proposals = json.loads(cache.read_text())
                else:
                    result = helper.predict(image, imgsz=args.proposal_imgsz, conf=0.4,
                                            device=device, verbose=False, max_det=100)[0]
                    proposals = polygons_from_result(result, semantic=False)
                    write_json(cache, proposals)
            if objects:
                label = args.output / "labels_draft" / (image_id + ".txt")
                label.write_text(yolo_text(objects), encoding="utf-8")
                record["draft_label"] = "labels_draft/" + label.name
            if proposals:
                proposal = args.output / "proposals" / (image_id + ".json")
                write_json(proposal, {"type": "unclassified_object_proposals", "reviewed": False,
                                     "folder_class_id_hint": expected_id,
                                     "note": "These masks may be background, whole leaves, or other objects. They are not disease labels.",
                                     "objects": proposals})
                record["proposal_file"] = "proposals/" + proposal.name
            status, reasons = classify_record(expected_id, objects, proposals, sha256 in conflict_hashes)
            record.update(status=status, reasons=reasons, objects=objects, proposal_count=len(proposals))
            preview = args.output / "previews" / (image_id + ".jpg")
            make_preview(image, objects, proposals, preview)
            record["preview"] = "previews/" + preview.name
        except Exception as exc:
            record.update(status="error", reasons=[type(exc).__name__ + ": " + str(exc)[:500]])
        db.execute("UPDATE images SET record=? WHERE id=?", (json.dumps(record, ensure_ascii=False), image_id))
        db.commit()
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            elapsed = time.monotonic() - started
            print(json.dumps({"processed": index + 1, "this_run_total": len(rows),
                              "elapsed_seconds": round(elapsed),
                              "seconds_per_image": round(elapsed / (index + 1), 2)}), flush=True)


def report(output, db):
    rows = db.execute("SELECT id,source,folder,expected_id,sha256,bytes,record FROM images ORDER BY folder,source").fetchall()
    records = []
    hashes = defaultdict(list)
    for image_id, source, folder, expected_id, sha, size, raw in rows:
        hashes[sha].append(image_id)
        record = json.loads(raw) if raw else {"id": image_id, "source": source, "folder": folder,
                  "folder_class_id": expected_id, "sha256": sha, "status": "pending",
                  "reviewed": False, "reasons": [], "objects": [], "proposal_count": 0}
        records.append(record)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            compact = {k: v for k, v in record.items() if k != "objects"}
            compact["semantic_object_count"] = len(record["objects"])
            compact["prediction_scores"] = [{"class_id": o["class_id"], "score": o["score"]}
                                               for o in record["objects"]]
            compact["duplicate_group_size"] = len(hashes[record["sha256"]])
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")
    fields = ["id", "folder", "folder_class_id", "status", "reviewed", "semantic_object_count",
              "proposal_count", "duplicate_group_size", "reasons", "source", "image", "draft_label", "proposal_file"]
    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            item["reasons"] = "; ".join(item["reasons"])
            writer.writerow({k: item.get(k) for k in fields})
    by_folder = {}
    for record in records:
        item = by_folder.setdefault(record["folder"], {"images": 0, "semantic_drafts": 0,
            "with_object_proposals": 0, "class_disagreements": 0, "errors": 0})
        item["images"] += 1
        item["semantic_drafts"] += bool(record["objects"])
        item["with_object_proposals"] += bool(record["proposal_count"])
        item["class_disagreements"] += "model_class_differs_from_folder" in record["reasons"]
        item["errors"] += record["status"] == "error"
    summary = {"total_images": len(records), "unique_file_contents": len(hashes),
               "status_counts": dict(Counter(r["status"] for r in records)),
               "folder_counts": dict(Counter(r["folder"] for r in records)),
               "all_labels_are_unreviewed": True,
               "class_conflict_groups": sum(len({r[3] for r in rows if r[4] == sha}) > 1 for sha, ids in hashes.items() if len(ids) > 1)}
    write_json(output / "summary.json", summary)
    write_json(output / "folder_summary.json", by_folder)
    write_json(output / "class_mapping.json", {"names": dict(enumerate(NAMES)),
               "old_model_ids_preserved": list(range(len(ORIGINAL_NAMES))),
               "new_class_ids_are_provisional": True, "folder_name_hints": FOLDERS})
    (output / "classes.yaml").write_text("names:\n" + "".join("  %d: %s\n" % (i,n) for i,n in enumerate(NAMES)), encoding="utf-8")
    write_json(output / "duplicate_groups.json", {sha: ids for sha, ids in hashes.items() if len(ids) > 1})
    (output / "README_TH.md").write_text(
        "# ชุด label ร่างจาก AI — ยังไม่ผ่านการยืนยัน\n\n"
        "งานนี้ไม่มีการเทรนหรือเปลี่ยน best.pt เดิม และไม่มีการตัดรูปซ้ำทิ้ง\n\n"
        "- images/: รูปสำหรับจับคู่กับ label; หมุนตาม EXIF และแปลงเป็น RGB JPEG แล้ว พิกัดตรงกับรูปในโฟลเดอร์นี้\n"
        "- labels_draft/: polygon ที่โมเดลโรคเดิมทำนาย หมายเลขคลาส 0–8 เดิม ไม่ใช่ ground truth\n"
        "- proposals/: ขอบเขตวัตถุจาก FastSAM ที่ยังไม่มีชื่อโรค ห้ามใช้เป็น label โรคโดยตรง\n"
        "- รูปที่ไม่มีไฟล์ใน labels_draft/ ไม่ได้หมายความว่าเป็นข้าวปกติ ให้ดู status ใน manifest\n"
        "- manifest.csv / manifest.jsonl: ตำแหน่งต้นฉบับ สถานะ ปัญหา และกลุ่มรูปซ้ำ\n"
        "- class_mapping.json / classes.yaml: หมายเลขคลาสเดิมคงเดิม คลาสใหม่เป็นข้อเสนอที่ต้องตกลงกับผู้เทรน\n"
        "- review.html: ดูภาพพร้อมขอบเขต กรองตามกลุ่มและสถานะได้\n\n"
        "ต้องตรวจความถูกต้องก่อนใช้เป็น ground truth โดยเฉพาะผลที่ชื่อโรคไม่ตรงกับโฟลเดอร์\n"
        "Healthy ต้องตกลงกับผู้เทรนว่าจะเป็นคลาสใหม่หรือภาพ background; ระบบไม่ตัดสินให้\n"
        "ก่อนแบ่ง train/val/test ให้เก็บรูปซ้ำและรูปดัดแปลงจากต้นฉบับเดียวกันไว้ในชุดเดียวกัน\n"
        "ไฟล์ label เป็น class_id ตามด้วยพิกัด polygon แบบ normalized ไม่มี confidence ต่อท้าย\n",
        encoding="utf-8")
    render_review(output, records, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def render_review(output, records, summary):
    translations = {
        "semantic_draft": "ร่างจากโมเดลโรค",
        "unclassified_proposals": "ขอบเขตวัตถุ — ยังไม่ระบุโรค",
        "needs_annotation": "ต้องเพิ่ม label",
        "pending": "รอประมวลผล",
        "error": "อ่านหรือประมวลผลไม่ได้",
        "unmapped_folder": "ยังไม่ได้จับคู่ชื่อโฟลเดอร์กับคลาส",
        "class_not_supported_by_old_model": "โมเดลเดิมยังไม่รู้จักกลุ่มนี้",
        "identical_content_in_different_classes": "ไฟล์เดียวกันอยู่ในโฟลเดอร์ต่างโรค",
        "model_class_differs_from_folder": "ชื่อโรคที่โมเดลทายไม่ตรงกับโฟลเดอร์",
        "low_model_score": "โมเดลให้คะแนนต่ำ ควรตรวจละเอียด",
        "no_semantic_label": "ยังไม่มีขอบเขตที่ระบุชื่อโรค",
        "unclassified_object_masks_are_not_disease_annotations": "เป็นขอบเขตวัตถุทั่วไป ต้องตรวจบริเวณอาการ",
        "confirm_healthy_class_vs_background_policy": "ต้องตกลงกับผู้เทรนว่าจะกำกับข้าวปกติอย่างไร",
    }
    cards = [{k: r.get(k) for k in ["id", "source", "folder", "status", "reasons", "preview", "image", "draft_label", "proposal_file"]} for r in records]
    data = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    template = """<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rice label drafts</title><style>
body{font:16px system-ui,sans-serif;margin:0;background:#f4f7f5;color:#16362b}header{padding:24px;position:sticky;top:0;background:#fff;border-bottom:1px solid #d5e3db;z-index:1}h1{font-size:25px;margin:0 0 8px}.notice{color:#945210}select,input,button{font:inherit;padding:9px;margin:10px 6px 0 0;border:1px solid #b9cec2;border-radius:8px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;padding:24px}article{background:white;border:1px solid #d5e3db;border-radius:12px;overflow:hidden}article img{width:100%;height:240px;object-fit:contain;background:#e9eeeb}article section{padding:14px}small{display:block;overflow-wrap:anywhere;color:#52665c;margin:7px 0}a{color:#126346;margin-right:10px}#more{margin:10px 24px 32px}#count{margin-left:12px}</style>
<header><h1>ตรวจ label ร่างจาก AI</h1><div class="notice">ทุกภาพยังไม่ผ่านการยืนยัน · ขอบเขต UNCLASSIFIED เป็นเพียงวัตถุที่ AI เสนอ ไม่ใช่การระบุโรค</div>
<select id="folder"><option value="">ทุกกลุ่ม</option></select><select id="status"><option value="">ทุกสถานะ</option></select><input id="search" placeholder="ค้นหาชื่อไฟล์หรือ ID"><span id="count"></span></header>
<main id="grid"></main><button id="more">แสดงเพิ่ม</button>
<script>const records=__DATA__;let filtered=records,shown=0;const $=id=>document.getElementById(id);
const labels=__LABELS__;
for(const key of ['folder','status'])for(const value of [...new Set(records.map(x=>x[key]))].sort()){const opt=document.createElement('option');opt.value=value;opt.textContent=labels[value]||value;$(key).append(opt)}
function link(s,label){const a=document.createElement('a');a.href=s;a.textContent=label;a.target='_blank';return a}
function add(){for(const r of filtered.slice(shown,shown+60)){const c=document.createElement('article');if(r.preview){const a=link(r.image,'');const im=document.createElement('img');im.src=r.preview;im.loading='lazy';im.alt=r.folder;a.append(im);c.append(a)}const section=document.createElement('section');const b=document.createElement('b');b.textContent=r.folder;section.append(b);for(const txt of [r.status,(r.source||r.id).split('/').pop(),...r.reasons]){const sm=document.createElement('small');sm.textContent=labels[txt]||txt;section.append(sm)}if(r.image)section.append(link(r.image,'รูปเต็ม'));if(r.draft_label)section.append(link(r.draft_label,'label ร่าง'));if(r.proposal_file)section.append(link(r.proposal_file,'ข้อเสนอวัตถุ'));c.append(section);$('grid').append(c)}shown=Math.min(shown+60,filtered.length);$('count').textContent=shown+' / '+filtered.length+' รูป';$('more').hidden=shown>=filtered.length}
function filter(){const q=$('search').value.toLowerCase();filtered=records.filter(r=>(!$('folder').value||r.folder===$('folder').value)&&(!$('status').value||r.status===$('status').value)&&((r.id+' '+r.source).toLowerCase().includes(q)));shown=0;$('grid').replaceChildren();add()}
for(const id of ['folder','status','search'])$(id).addEventListener('input',filter);$('more').onclick=add;filter();</script></html>"""
    (output / "review.html").write_text(template.replace("__DATA__", data).replace("__LABELS__", json.dumps(translations)), encoding="utf-8")


def package(output, include_images):
    name = output.name + ("_with_images.zip" if include_images else "_labels_metadata.zip")
    target = output.parent / name
    dirs = ["labels_draft", "proposals"] + (["images", "previews"] if include_images else [])
    with zipfile.ZipFile(target, "w", allowZip64=True) as z:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".yaml", ".html", ".md"}:
                z.write(p, p.name, compress_type=zipfile.ZIP_DEFLATED)
        for directory in dirs:
            for p in sorted((output / directory).glob("*")):
                z.write(p, str(p.relative_to(output)), compress_type=zipfile.ZIP_STORED if p.suffix==".jpg" else zipfile.ZIP_DEFLATED)
    print("Package: " + str(target.resolve()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "run", "report", "package"])
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("best.pt"))
    parser.add_argument("--proposal-model", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--proposal-imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--retry-unlabeled", action="store_true",
                        help="Also process records with no semantic polygons or object proposals")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")
    with open_db(args.output) as db:
        if args.source:
            print("New inventory entries: " + str(discover(args.source, db)), flush=True)
        if args.action == "run":
            run(args, db)
        if args.action == "package":
            report(args.output, db)
            package(args.output, args.include_images)
        else:
            report(args.output, db)


if __name__ == "__main__":
    main()
