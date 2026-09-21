"""Exploratory sanity audit for the frozen Fitzpatrick phenotype instrument.

This does not replace any confirmatory metric.  It retrains the existing TinyCNN recipe
with an explicit torch seed, then reports class-conditional behaviour on real held-out
source- and target-population images from the frozen P1/P4 manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.fitzpatrick import load_fitzpatrick
from evaluation.fitz_instruments import _to_float, train_phenotype_instrument


def hashes_for(condition: dict, keys: tuple[str, ...]) -> list[str]:
    return [h for key in keys for h in condition.get(key, [])]


@torch.no_grad()
def target_share(model, raw, hashes: list[str], device: torch.device) -> float:
    idx = raw.indices(hashes)
    x = _to_float(raw.images, idx, device)
    hits = 0
    for start in range(0, len(hashes), 256):
        hits += int((model(x[start:start + 256]).argmax(1) == 1).sum())
    return hits / len(hashes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = []

    for manifest_path in args.manifests:
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        conditions = manifest["conditions"]
        all_hashes = sorted({h for c in conditions.values() for key, values in c.items()
                             if isinstance(values, list) for h in values})
        test_names = manifest["split"]["test"]
        test_hashes = {h for name in test_names
                       for h in hashes_for(conditions[name], (
                           "src_support", "src_query", "tgt_support_reserve", "tgt_query"))}
        exclude = {h for c in conditions.values() for h in c.get("tgt_query", [])}
        raw = load_fitzpatrick(image_size=32, hashes=all_hashes, root=args.root,
                               verbose=False, label_field=manifest["task_field"])

        # The confirmatory evaluator did not explicitly seed torch before TinyCNN
        # construction.  This diagnostic does, solely to make the sanity check repeatable.
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        source_types = tuple(manifest["config"].get("source_phototypes", [1, 2, 3]))
        target_types = tuple(manifest["config"].get("target_phototypes", [4, 5, 6]))
        rep = train_phenotype_instrument(
            raw, exclude=exclude, test_hashes=test_hashes, device=device, seed=args.seed,
            source_phototypes=source_types, target_phototypes=target_types, verbose=False)

        per_family = []
        src_n = tgt_n = 0
        src_pos_weighted = tgt_pos_weighted = 0.0
        for name in test_names:
            c = conditions[name]
            src = hashes_for(c, ("src_support", "src_query"))
            tgt = hashes_for(c, ("tgt_support_reserve", "tgt_query"))
            src_pos = target_share(rep.model, raw, src, device)
            tgt_pos = target_share(rep.model, raw, tgt, device)
            per_family.append({"family": name, "source_n": len(src), "target_n": len(tgt),
                               "real_source_target_share": src_pos,
                               "real_target_target_share": tgt_pos})
            src_n += len(src); tgt_n += len(tgt)
            src_pos_weighted += src_pos * len(src)
            tgt_pos_weighted += tgt_pos * len(tgt)

        src_pos = src_pos_weighted / src_n
        tgt_pos = tgt_pos_weighted / tgt_n
        rows.append({
            "manifest": path.as_posix(),
            "test_families": test_names,
            "source_phototypes": source_types,
            "target_phototypes": target_types,
            "reported_overall_test_accuracy": rep.accuracy.get(
                "test conditions (the transfer it is used for)"),
            "real_source_target_share": src_pos,
            "real_target_target_share": tgt_pos,
            "real_source_recall": 1.0 - src_pos,
            "real_target_recall": tgt_pos,
            "balanced_accuracy": 0.5 * ((1.0 - src_pos) + tgt_pos),
            "source_n": src_n,
            "target_n": tgt_n,
            "per_family": per_family,
        })
        del rep, raw
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({"status": "EXPLORATORY", "torch_seed": args.seed, "rows": rows},
                     indent=2))


if __name__ == "__main__":
    main()
