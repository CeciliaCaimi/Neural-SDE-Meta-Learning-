"""What a task split would actually allocate, from the images now on disk.

Allocation follows the CIFAR builder: the target side of a condition reserves
max(k_shots) = 20 images as the support reserve, from which K_T draws a nested prefix, and
the query takes everything that remains. The source side takes M_S support images, plus a
query batch **only for conditions used in training**: src_query feeds the source denoising
term of the meta objective, and nothing at meta-test reads it. Evaluation touches
src_support, tgt_support and tgt_query and no other stream.

So a training task and a held-out task have different requirements, and counting them
alike understates how many conditions are usable.
"""
from __future__ import annotations

import collections
import csv
import os

ROOT = os.environ.get("FITZPATRICK_ROOT", r"C:\T1D Meta Learning\dataset\fitzpatrick17k")
KT_MAX = 20          # tgt_support_reserve = max(k_shots)
TGT_QUERY_MIN = 25   # the smallest held-out target set worth reporting a distance on


def side(scale: str) -> str | None:
    try:
        v = int(scale)
    except ValueError:
        return None
    return "source" if 1 <= v <= 3 else ("target" if 4 <= v <= 6 else None)


def load() -> tuple[collections.Counter, collections.Counter]:
    got = set()
    with open(os.path.join(ROOT, "manifest.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "cached"):
                got.add(r["md5hash"])
    src, tgt = collections.Counter(), collections.Counter()
    with open(os.path.join(ROOT, "fitzpatrick17k.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = side(r["fitzpatrick_scale"])
            if s and r["md5hash"] in got:
                (src if s == "source" else tgt)[r["label"]] += 1
    return src, tgt


def main() -> None:
    src, tgt = load()
    labels = sorted(set(src) | set(tgt))

    print("All counts are of files on disk.\n")
    print(f"A held-out task needs  M_S source + {KT_MAX} target reserve + "
          f"{TGT_QUERY_MIN} held-out target.")
    print(f"A training task needs  M_S + src_query source + {KT_MAX} target reserve + "
          f"a target query.\n")

    print(f"{'M_S':>5}{'train-capable':>16}{'eval-capable':>15}   (src_query = 32 for training)")
    print("-" * 62)
    for m_s in (64, 32, 16, 8):
        trainable = [l for l in labels
                     if src[l] >= m_s + 32 and tgt[l] >= KT_MAX + TGT_QUERY_MIN]
        evaluable = [l for l in labels
                     if src[l] >= m_s and tgt[l] >= KT_MAX + TGT_QUERY_MIN]
        print(f"{m_s:>5}{len(trainable):>16}{len(evaluable):>15}")
    print()

    m_s = 16
    evaluable = sorted([l for l in labels
                        if src[l] >= m_s and tgt[l] >= KT_MAX + TGT_QUERY_MIN],
                       key=lambda l: -tgt[l])
    trainable = {l for l in evaluable if src[l] >= m_s + 32}

    print("=" * 78)
    print(f"Per-condition allocation at M_S = {m_s}, target reserve {KT_MAX}")
    print("=" * 78)
    print(f"  {'condition':<30}{'src pool':>9}{'tgt pool':>9}{'reserve':>9}"
          f"{'held out':>10}{'train?':>8}")
    print("  " + "-" * 76)
    held = {}
    for l in evaluable:
        held[l] = tgt[l] - KT_MAX
        print(f"  {l[:30]:<30}{src[l]:>9}{tgt[l]:>9}{KT_MAX:>9}{held[l]:>10}"
              f"{'yes' if l in trainable else 'no':>8}")
    print("  " + "-" * 76)
    print(f"  {'total':<30}{sum(src[l] for l in evaluable):>9}"
          f"{sum(tgt[l] for l in evaluable):>9}{KT_MAX * len(evaluable):>9}"
          f"{sum(held.values()):>10}")
    print()

    n = len(evaluable)
    print(f"{n} conditions are usable as held-out tasks; {len(trainable)} of them can also "
          f"carry a training\nsource query batch. The plan splits the semantic tasks "
          f"themselves, so the split is over these.\n")

    for a, b, c in ((2, 2, 2), (3, 2, 2), (2, 1, 3)):
        if a + b + c != n:
            continue
        tr, va, te = evaluable[:a], evaluable[a:a + b], evaluable[a + b:]
        print(f"  split {a} / {b} / {c}")
        for part, names, is_train in (("train", tr, True), ("val", va, False), ("test", te, False)):
            s_img = sum(m_s + (32 if is_train else 0) for _ in names)
            r_img = KT_MAX * len(names)
            h_img = sum(held[l] for l in names)
            flag = "" if not is_train else ("  <- " + ", ".join(
                l for l in names if l not in trainable) + " cannot fund src_query"
                if any(l not in trainable for l in names) else "")
            print(f"    {part:<6}{len(names):>2} tasks   source {s_img:>4}   "
                  f"reserve {r_img:>3}   held-out target {h_img:>4}{flag}")
        print()


if __name__ == "__main__":
    main()
