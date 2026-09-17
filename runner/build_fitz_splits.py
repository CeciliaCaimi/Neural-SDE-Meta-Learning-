"""Build the Fitzpatrick17k task split: train / validation / test over CONDITIONS.

    python -m runner.build_fitz_splits --out artifacts/fitzpatrick_split.json

What is being split is the semantic task itself -- the skin condition -- not the images
within a task. That is the whole point of stage C: if conditions were shared between
train and test, a model could learn each condition's coordinate and the relation would
never be the object of generalisation. Held-out conditions are never seen in training.

The relation is the population shift. Source is Fitzpatrick skin type I-III, target is
IV-VI, chosen in C0 from the observed counts rather than in advance: the I-II vs V-VI
grouping leaves three usable conditions on the images that are reachable, which cannot be
split into train, validation and test at all.

Unlike CIFAR, where the target domain is a corruption applied to a source image at fetch
time, here source and target are **different images of different people**. The two streams
are disjoint by construction, since an image carries one skin-type annotation.

Allocation per condition, following the CIFAR builder:

  tgt_support_reserve   max(k_shots) = 20 images; K_T draws a nested prefix of this
  tgt_query             everything left on the target side, held out, >= 25
  src_support           the source pool, from which M_S draws a nested prefix
  src_query             32 images, disjoint from src_support -- TRAINING CONDITIONS ONLY

That last line is what constrains the split. A training condition feeds the source
denoising term of the meta objective and so needs M_S + 32 source images; an evaluation
condition needs only M_S, because nothing at meta-test reads src_query. A4 found the
transported coordinate already saturated at M_S = 16 on CIFAR, which is what makes a
16-image source support defensible rather than a compromise.

The dataset is never committed and never copied into the tree. Point FITZPATRICK_ROOT at
your own copy; the split file stores md5 hashes, not paths, so it is machine-independent.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_ROOT = os.environ.get(
    "FITZPATRICK_ROOT", r"C:\T1D Meta Learning\dataset\fitzpatrick17k")

SEED = 20260917
M_S = 16              # source support drawn on at meta-test; A4 says this is enough
SRC_QUERY = 32        # source denoising batch, training conditions only
KT_MAX = 20           # tgt_support_reserve = max(k_shots); overridable on the command line
TGT_QUERY_MIN = 25    # smallest held-out target set worth reporting a distance on


def side(scale: str) -> str | None:
    """Fitzpatrick I-III is the source population, IV-VI the target. -1 is unlabelled."""
    try:
        v = int(scale)
    except (TypeError, ValueError):
        return None
    if 1 <= v <= 3:
        return "source"
    if 4 <= v <= 6:
        return "target"
    return None


def near_duplicate_groups(root: str) -> list[list[str]]:
    """Groups of near-identical images, from CleanPatrick's human annotations.

    near_duplicates.csv carries one row per candidate group with a label: 1 for a group
    judged near-identical, 0 for one judged distinct. Only the 1s are used; the 0s are
    the annotators saying these images are genuinely different, and honouring them is the
    difference between removing duplicates and removing data.

    Fitzpatrick17k is assembled from atlases that republish the same photograph under
    different entries, so this is not a hypothetical. A group that straddles two
    conditions makes a held-out task partly seen; one that straddles the source and
    target sides makes the population shift partly an identity map; one that straddles
    the support reserve and the query makes the held-out query partly the support.
    """
    path = os.path.join(root, "near_duplicates.csv")
    groups = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["label"].strip() == "1":
                members = [h.strip() for h in r["md5hash_list"].split(",") if h.strip()]
                if len(members) > 1:
                    groups.append(members)
    return groups


def components(groups: list[list[str]]) -> dict[str, str]:
    """Union-find over the groups: a hash can appear in several, and the transitive
    closure is what has to be collapsed, not the individual rows."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        first = find(g[0])
        for h in g[1:]:
            parent[find(h)] = first
    return {h: find(h) for h in parent}


def load(root: str, dedup: bool = True) -> tuple[dict[str, dict[str, list[str]]], dict]:
    """Condition -> {'source': [md5...], 'target': [md5...]}, files on disk only."""
    got = set()
    with open(os.path.join(root, "manifest.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "cached"):
                got.add(r["md5hash"])

    # every on-disk image with a usable phototype, with where it would belong
    placed: dict[str, tuple[str, str]] = {}
    with open(os.path.join(root, "fitzpatrick17k.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = side(r["fitzpatrick_scale"])
            if s and r["md5hash"] in got:
                placed[r["md5hash"]] = (r["label"], s)

    stats = {"images_before": len(placed), "dropped": 0,
             "groups_on_disk": 0, "groups_across_conditions": 0,
             "groups_across_domains": 0, "dedup": dedup}

    if dedup:
        root_of = components(near_duplicate_groups(root))
        members: dict[str, list[str]] = collections.defaultdict(list)
        for h in placed:
            if h in root_of:
                members[root_of[h]].append(h)

        keep_one = set()
        for comp, hs in members.items():
            if len(hs) < 2:
                continue
            stats["groups_on_disk"] += 1
            conds = {placed[h][0] for h in hs}
            sides = {placed[h][1] for h in hs}
            if len(conds) > 1:
                stats["groups_across_conditions"] += 1
            if len(sides) > 1:
                stats["groups_across_domains"] += 1
            # keep the lexicographically first, drop the rest: deterministic, and it does
            # not let the choice of representative depend on anything downstream
            keep_one.add(min(hs))
            for h in hs:
                if h != min(hs):
                    placed.pop(h)
                    stats["dropped"] += 1

    pools: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: {"source": [], "target": []})
    for h, (label, s) in placed.items():
        pools[label][s].append(h)
    stats["images_after"] = len(placed)
    return pools, stats


def assign(pools, m_s: int, kt_max: int, min_query: int = TGT_QUERY_MIN) -> tuple[list[str], list[str], list[str]]:
    """Which conditions go to train, validation and test.

    The assignment is forced in one direction and balanced in the other, and both rules
    are fixed here rather than chosen after seeing any result:

      1. A condition that cannot fund M_S + SRC_QUERY source images cannot be a training
         condition. Every condition that can, is one -- with so few to go round, holding
         one back would cost more than it buys.
      2. The rest are dealt to validation and test by giving the next-largest condition to
         whichever side currently holds fewer held-out target images, **subject to the two
         sides holding an equal number of conditions**. Balancing on images alone put four
         conditions in validation and two in test, and a two-condition test set cannot
         carry a claim about generalising to unseen tasks whatever its image count. The
         count is the binding quantity here and the image balance is the tie-break.

    Neither rule looks at any model output; no model has been run on this dataset.
    """
    usable = [c for c, p in pools.items()
              if len(p["source"]) >= m_s and len(p["target"]) >= kt_max + min_query]
    train = sorted(c for c in usable if len(pools[c]["source"]) >= m_s + SRC_QUERY)
    rest = sorted((c for c in usable if c not in set(train)),
                  key=lambda c: -len(pools[c]["target"]))

    val: list[str] = []
    test: list[str] = []
    val_imgs = test_imgs = 0
    cap = (len(rest) + 1) // 2          # neither side may take more than half
    for c in rest:
        held = len(pools[c]["target"]) - kt_max
        to_val = val_imgs < test_imgs
        if to_val and len(val) >= cap:
            to_val = False
        elif not to_val and len(test) >= cap:
            to_val = True
        if to_val:
            val.append(c)
            val_imgs += held
        else:
            test.append(c)
            test_imgs += held
    return train, sorted(val), sorted(test)


def build(root: str, m_s: int, kt_max: int, dedup: bool = True,
          min_query: int = TGT_QUERY_MIN) -> dict:
    pools, stats = load(root, dedup)
    train, val, test = assign(pools, m_s, kt_max, min_query)
    rng = random.Random(SEED)

    conditions = {}
    for part, names in (("train", train), ("val", val), ("test", test)):
        for c in names:
            src = sorted(pools[c]["source"])
            tgt = sorted(pools[c]["target"])
            rng.shuffle(src)
            rng.shuffle(tgt)
            if part == "train":
                src_query, src_support = src[:SRC_QUERY], src[SRC_QUERY:]
            else:
                src_query, src_support = [], src
            conditions[c] = {
                "split": part,
                "src_support": src_support,
                "src_query": src_query,
                "tgt_support_reserve": tgt[:kt_max],
                "tgt_query": tgt[kt_max:],
            }

    ids = sorted(h for c in conditions.values()
                 for stream in ("src_support", "src_query", "tgt_support_reserve", "tgt_query")
                 for h in c[stream])
    if len(ids) != len(set(ids)):
        raise SystemExit("an image was allocated to more than one stream")

    return {
        "builder": "build_fitz_splits.py",
        "dataset": "fitzpatrick17k",
        "relation": {"name": "phenotype", "source": "Fitzpatrick I-III",
                     "target": "Fitzpatrick IV-VI"},
        "config": {"seed": SEED, "m_source": m_s, "src_query": SRC_QUERY,
                   "tgt_support_reserve": kt_max, "tgt_query_min": min_query},
        "split": {"train": train, "val": val, "test": test},
        "conditions": conditions,
        "dedup": stats,
        "n_images": len(ids),
        "checksum": hashlib.sha256("".join(ids).encode()).hexdigest(),
    }


def report(payload: dict) -> None:
    cfg = payload["config"]
    print(f"relation: {payload['relation']['source']}  ->  {payload['relation']['target']}")
    print(f"M_S={cfg['m_source']}  src_query={cfg['src_query']}  "
          f"reserve={cfg['tgt_support_reserve']}  seed={cfg['seed']}\n")
    print(f"  {'condition':<34}{'split':>7}{'src sup':>9}{'src qry':>9}"
          f"{'reserve':>9}{'tgt qry':>9}")
    print("  " + "-" * 77)
    for part in ("train", "val", "test"):
        for c in payload["split"][part]:
            d = payload["conditions"][c]
            print(f"  {c[:34]:<34}{part:>7}{len(d['src_support']):>9}"
                  f"{len(d['src_query']):>9}{len(d['tgt_support_reserve']):>9}"
                  f"{len(d['tgt_query']):>9}")
        print("  " + "-" * 77)
    for part in ("train", "val", "test"):
        names = payload["split"][part]
        held = sum(len(payload["conditions"][c]["tgt_query"]) for c in names)
        src = sum(len(payload["conditions"][c]["src_support"]) for c in names)
        print(f"  {part:<7}{len(names):>3} conditions   {src:>5} source support   "
              f"{held:>5} held-out target")
    print(f"\n  {payload['n_images']} images allocated, no image in two streams")
    print(f"  checksum {payload['checksum']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--m-source", type=int, default=M_S)
    ap.add_argument("--max-kt", type=int, default=KT_MAX,
                    help="the support reserve = max(k_shots). Only the maximum costs "
                         "images: K_T draws a nested prefix of the reserve.")
    ap.add_argument("--min-query", type=int, default=TGT_QUERY_MIN,
                    help="smallest held-out target query set a condition may have. "
                         "It is a judgement, not a measurement; whatever it is set to "
                         "is recorded in the split file.")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip near-duplicate removal. Only for measuring what it "
                         "costs; a split built this way leaks between tasks.")
    ap.add_argument("--out", default="artifacts/fitzpatrick_split.json")
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
    if os.path.exists(out):
        raise SystemExit(
            f"{out} already exists. A split file is the shared input of every downstream "
            "run, so this refuses to overwrite one. Pass --out with a new name, or delete "
            "that file deliberately if you mean to rebuild it.")

    payload = build(a.root, a.m_source, a.max_kt, not a.no_dedup, a.min_query)
    report(payload)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"\nwritten to {os.path.relpath(out, _ROOT)}")


if __name__ == "__main__":
    main()
