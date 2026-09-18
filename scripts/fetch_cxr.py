r"""Fetch NIH ChestX-ray14 into CHESTXRAY_ROOT. Resumable; re-running costs nothing.

    python scripts/fetch_cxr.py                # everything
    python scripts/fetch_cxr.py --only csv     # just the annotation tables

Source: the Hugging Face mirror of the official NIH release,
`alkzar90/NIH-Chest-X-ray-dataset`, which carries the same twelve image archives and the
official `Data_Entry_2017_v2020.csv`. It is used instead of nihcc.app.box.com because Box
returns HTTP 403 to unauthenticated programmatic requests, so the official host cannot be
scripted; the bytes are the same release. Say "a mirror of the official NIH release" in any
write-up, not "the official release".

**The archives are not extracted.** Forty-five gigabytes of zips would become ninety, and the
split needs a few thousand images of the 112 120. domains/chestxray.py reads PNGs straight out
of the zips through an index built from their central directories, exactly as
domains/fitzpatrick.py decodes on demand.

Nothing is written inside the repository. The directory holding the data on the machine of
record also holds clinical data, so this only ever writes into CHESTXRAY_ROOT itself.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

REPO = "alkzar90/NIH-Chest-X-ray-dataset"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
DEFAULT_ROOT = os.environ.get("CHESTXRAY_ROOT") or os.path.normpath(
    os.path.join(_ROOT, "..", "..", "..", "dataset", "chestxray14"))

CSVS = ["data/Data_Entry_2017_v2020.csv", "data/BBox_List_2017.csv"]
ZIPS = [f"data/images/images_{i:03d}.zip" for i in range(1, 13)]


def remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def fetch(path: str, root: str, retries: int = 5) -> tuple[str, int]:
    """Download one file, resuming a partial one with a Range request."""
    url = BASE + path
    dst = os.path.join(root, os.path.basename(path))
    want = remote_size(url)
    have = os.path.getsize(dst) if os.path.exists(dst) else 0

    if want is not None and have == want:
        return "cached", have
    if want is not None and have > want:          # truncated remote or a stale file
        os.remove(dst)
        have = 0

    for attempt in range(retries):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            t0, last = time.time(), have
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(dst, "ab" if have else "wb") as f:
                while True:
                    chunk = r.read(1 << 22)       # 4 MiB
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if have - last > (1 << 28):   # progress every 256 MiB
                        rate = (have - last) / max(1e-9, time.time() - t0) / 1e6
                        pct = f"{100*have/want:.0f}%" if want else "?"
                        print(f"    {os.path.basename(path)}  {have/1e9:.2f} GB  {pct}  "
                              f"{rate:.0f} MB/s", flush=True)
                        t0, last = time.time(), have
            if want is None or have == want:
                return "ok", have
            print(f"    short read ({have} of {want}), resuming", flush=True)
        except Exception as e:
            print(f"    attempt {attempt+1}/{retries} failed: {type(e).__name__}: "
                  f"{str(e)[:80]}", flush=True)
            time.sleep(5 * (attempt + 1))
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
    return "failed", have


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--only", choices=("csv", "zips", "all"), default="all")
    a = ap.parse_args()

    os.makedirs(a.root, exist_ok=True)
    want = (CSVS if a.only == "csv" else ZIPS if a.only == "zips" else CSVS + ZIPS)

    print(f"root   {a.root}")
    print(f"source {REPO}")
    print(f"files  {len(want)}\n")

    rows, total, t_start = [], 0, time.time()
    for i, path in enumerate(want, 1):
        print(f"[{i}/{len(want)}] {path}", flush=True)
        status, n = fetch(path, a.root)
        total += n
        rows.append({"path": path, "file": os.path.basename(path),
                     "status": status, "bytes": n})
        print(f"    {status}  {n/1e9:.2f} GB", flush=True)

    with open(os.path.join(a.root, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "file", "status", "bytes"])
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r["status"] == "failed"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} files, {total/1e9:.2f} GB, "
          f"{(time.time()-t_start)/60:.0f} min")
    if bad:
        print("FAILED: " + ", ".join(r["file"] for r in bad))
        print("Re-run; completed files are skipped and partial ones resume.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
