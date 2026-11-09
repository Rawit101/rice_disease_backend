"""Restore normalized coordinate-reference images from the original archive.

Used with the small labels-only handoff. Does not modify original source files.
SHA256 verification runs before conversion. Missing detections remain unlabeled.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

from PIL import Image,ImageOps


def within(root, relative):
    path=(root/relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Path escapes the selected source/output directory: '+relative)
    return path


def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def restore(source,root,workers=4):
    source=Path(source).resolve();root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    by_sha=defaultdict(list)
    for r in manifest:
        if r['image'] and r['quality']!='read_error':
            by_sha[r['sha256']].append(r)
    def convert(rs):
        candidates=[within(source,r['source_relative_path']) for r in rs]
        original=next((p for p in candidates if p.is_file()),None)
        if original is None:
            raise FileNotFoundError('Missing original: '+rs[0]['source_relative_path'])
        if checksum(original)!=rs[0]['sha256']:
            raise ValueError('Original checksum differs: '+str(original))
        with Image.open(original) as im:
            im.load(); im=ImageOps.exif_transpose(im).convert('RGB')
        r=rs[0]
        if (r.get('width'),r.get('height'))!=(im.width,im.height):
            raise ValueError('Normalized dimensions differ: '+str(original))
        destination=within(root,r['image']);destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            temporary=destination.with_suffix('.restoring.jpg')
            im.save(temporary,quality=95);temporary.replace(destination)
        thumb=im.copy();thumb.thumbnail((384,300))
        for r in rs:
            copy=within(root,r['image']);copy.parent.mkdir(parents=True,exist_ok=True)
            if not copy.exists():
                try:os.link(destination,copy)
                except OSError:shutil.copyfile(destination,copy)
            preview=within(root,r['thumbnail']);preview.parent.mkdir(parents=True,exist_ok=True)
            if not preview.exists():thumb.save(preview,quality=82)
        return len(rs)
    count=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,n in enumerate(pool.map(convert,by_sha.values()),1):
            count+=n
            if i%500==0:
                print(json.dumps(dict(unique_restored=i,unique_total=len(by_sha),source_files_restored=count)),flush=True)
    result=dict(restored_source_files=count,unique_verified_sources=len(by_sha),
                originals_modified=False,exif_orientation_applied=True)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Extracted Rice Disease directory')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();restore(args.source,args.root,args.workers)
