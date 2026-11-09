"""Download a user-shared Drive ZIP without credentials and extract it safely.

Only follows Drive's public download confirmation form. No browser cookies or
Google credentials are read. Interrupted downloads can be resumed when the
server supports HTTP Range requests.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import stat
import time
import zipfile

import requests


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.inputs = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
            self.active = True
        if tag == "input" and self.active and attrs.get("name"):
            self.inputs[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def download(file_id, destination, expected_size):
    if destination.exists() and destination.stat().st_size == expected_size:
        print("Archive already downloaded", flush=True)
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id},
        stream=True,
        timeout=(30, 120),
    )
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        form = DownloadForm()
        form.feed(response.text)
        response.close()
        if form.action != "https://drive.usercontent.google.com/download":
            raise RuntimeError("Drive did not return a public download form")
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": "bytes=%d-" % offset} if offset else {}
        response = session.get(form.action, params=form.inputs, headers=headers,
                               stream=True, timeout=(30, 120))
        response.raise_for_status()
        if "text/html" in response.headers.get("content-type", ""):
            raise RuntimeError("Drive returned a page instead of the ZIP (access or quota limit)")
        if response.status_code == 206:
            if not response.headers.get("Content-Range", "").startswith("bytes %d-" % offset):
                raise RuntimeError("Unexpected download range")
        else:
            offset = 0
    else:
        offset = 0
    started = time.monotonic()
    last = started
    total = offset
    with response, partial.open("ab" if offset else "wb") as out:
        for chunk in response.iter_content(4 * 1024 * 1024):
            if not chunk:
                continue
            out.write(chunk)
            total += len(chunk)
            if time.monotonic() - last >= 5:
                elapsed = time.monotonic() - started
                print(json.dumps({"downloaded_mb": round(total / 1e6),
                                  "percent": round(total / expected_size * 100, 1),
                                  "mb_per_second": round((total - offset) / elapsed / 1e6, 1)}), flush=True)
                last = time.monotonic()
    if partial.stat().st_size != expected_size:
        raise RuntimeError("Download size differs from Drive metadata")
    if not zipfile.is_zipfile(partial):
        raise RuntimeError("Downloaded file is not a ZIP")
    partial.replace(destination)


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        print(json.dumps({"archive_members": len(members),
                          "uncompressed_gb": round(sum(x.file_size for x in members) / 1e9, 2)}), flush=True)
        last = time.monotonic()
        for index, member in enumerate(members):
            target = (root / member.filename).resolve()
            if root not in target.parents or stat.S_ISLNK(member.external_attr >> 16):
                raise RuntimeError("Unsafe ZIP member")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with z.open(member) as src, temporary.open("wb") as out:
                while True:
                    chunk = src.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            temporary.replace(target)
            if time.monotonic() - last >= 5:
                print(json.dumps({"extracted": index + 1, "total": len(members)}), flush=True)
                last = time.monotonic()
    print("Extraction complete: " + str(root), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--expected-size", required=True, type=int)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--extract-to", required=True, type=Path)
    args = parser.parse_args()
    download(args.file_id, args.archive, args.expected_size)
    extract(args.archive, args.extract_to)
