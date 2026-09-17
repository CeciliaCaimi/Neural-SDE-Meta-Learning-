"""Fetch the retrievable part of Fitzpatrick17k and record exactly what arrived.

    python fetch_fitz.py --limit 40          # pilot
    python fetch_fitz.py                     # everything reachable

Writes images under DEST/images/<md5hash>.<ext> and a manifest beside them. Resumable: a
file already on disk with plausible image bytes is not fetched again.

A row counts as retrieved only if the response is 2xx AND the bytes carry an image magic
number. A 200 that returns an HTML error page is a dead link wearing a disguise, and
counting it would inflate every table built on top of this.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

UA = "Mozilla/5.0 (research dataset retrieval)"

DEFAULT_ROOT = os.environ.get("FITZPATRICK_ROOT", r"C:\T1D Meta Learning\dataset\fitzpatrick17k")
CSV = os.path.join(DEFAULT_ROOT, "fitzpatrick17k.csv")
DEST = DEFAULT_ROOT

MAGIC = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"BM": "bmp",
}


def sniff(buf: bytes) -> str | None:
    for magic, ext in MAGIC.items():
        if buf.startswith(magic):
            return ext
    if buf[:4] == b"RIFF" and buf[8:12] == b"WEBP":
        return "webp"
    return None


lock = threading.Lock()
counts = {"ok": 0, "cached": 0, "dead": 0, "error": 0}


def existing(md5: str) -> str | None:
    for ext in ("jpg", "png", "gif", "bmp", "webp"):
        p = os.path.join(DEST, "images", f"{md5}.{ext}")
        if os.path.exists(p) and os.path.getsize(p) > 512:
            return p
    return None


def fetch(row: dict, session=None, retries: int = 3) -> dict:
    md5, url = row["md5hash"], row["url"]
    rec = {"md5hash": md5, "label": row["label"], "fitzpatrick_scale": row["fitzpatrick_scale"],
           "host": urlparse(url).netloc, "url": url, "status": "", "bytes": 0, "path": ""}

    hit = existing(md5)
    if hit:
        rec.update(status="cached", bytes=os.path.getsize(hit), path=hit)
        with lock:
            counts["cached"] += 1
        return rec

    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as resp:
                body = resp.read()
            ext = sniff(body[:16])
            if ext is None:
                last = "not_an_image"
                break
            path = os.path.join(DEST, "images", f"{md5}.{ext}")
            with open(path, "wb") as f:
                f.write(body)
            rec.update(status="ok", bytes=len(body), path=path)
            with lock:
                counts["ok"] += 1
            return rec
        except urllib.error.HTTPError as e:
            last = f"http_{e.code}"
            if e.code in (404, 410):
                break
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            last = f"{type(e).__name__}"
            time.sleep(1.5 * (attempt + 1))

    rec["status"] = last or "unknown"
    with lock:
        counts["dead" if last in ("not_an_image",) or last.startswith("http_") else "error"] += 1
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="sample this many rows instead of all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--host", default="atlasdermatologico.com.br",
                    help="only fetch rows served by this host; empty string means every host")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()

    os.makedirs(os.path.join(DEST, "images"), exist_ok=True)

    with open(CSV, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["url"]]
    if a.host:
        rows = [r for r in rows if urlparse(r["url"]).netloc == a.host]
    if a.limit:
        random.Random(20260917).shuffle(rows)
        rows = rows[: a.limit]

    print(f"{len(rows)} rows to fetch from {a.host or 'every host'}", flush=True)

    session = None

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, rec in enumerate(pool.map(lambda r: fetch(r, session), rows), 1):
            out.append(rec)
            if i % 200 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"  {i}/{len(rows)}  ok={counts['ok']} cached={counts['cached']} "
                      f"dead={counts['dead']} error={counts['error']}  {el:.0f}s", flush=True)

    manifest = a.manifest or os.path.join(DEST, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    got = [r for r in out if r["status"] in ("ok", "cached")]
    print()
    print(f"retrieved {len(got)} / {len(rows)} ({100 * len(got) / max(1, len(rows)):.1f}%)")
    if got:
        mb = sum(r["bytes"] for r in got) / 1e6
        print(f"bytes {mb:.1f} MB, mean {mb * 1000 / len(got):.0f} KB per image")
    fail = {}
    for r in out:
        if r["status"] not in ("ok", "cached"):
            fail[r["status"]] = fail.get(r["status"], 0) + 1
    if fail:
        print("failures:", dict(sorted(fail.items(), key=lambda kv: -kv[1])))
    print(f"manifest written to {manifest}")


if __name__ == "__main__":
    sys.exit(main())
