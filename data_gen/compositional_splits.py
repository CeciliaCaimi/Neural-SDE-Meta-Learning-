# data_gen/compositional_splits.py
#
# Item 6, phase 2: deterministic compositional-generalisation splits over
# the item-5 factorial drift x diffusion benchmark.
#
# Unit of holdout: a (drift_mechanism, diffusion_mechanism) PAIR -- there
# are 14 drift mechanisms x 8 diffusion mechanisms = 112 pairs, each
# containing 16 tasks (4 independently-sampled drift coefficient draws x 4
# independently-sampled diffusion coefficient draws, from
# data_gen/generate_factorial_benchmark.py's default
# n_drift_per_mechanism=4 / n_diffusion_per_mechanism=4). All 16 tasks of a
# held-out pair are test tasks, in the "heldout" (compositional,
# recombination) test set, and never appear in training in any role.
#
# Coverage constraint: holding out a PAIR removes that (drift, diffusion)
# combination, but every individual drift mechanism and every individual
# diffusion mechanism must still appear in at least one TRAINING pair --
# otherwise the eval would be testing generalisation to an unseen
# mechanism, not recombination of previously-seen mechanisms (the point of
# item 6). The selection below enforces this directly (never holds out a
# pair that would zero out its row or column) and additionally tries to
# balance holdout counts evenly across rows and across columns (greedily
# picking the most under-held row/column at each step), so held-out pairs
# don't cluster onto a handful of mechanisms.
#
# In-distribution (ID) test set: for every TRAINING pair, 2 of its 16 tasks
# (deterministic given the seed) are set aside as "id_test" -- never used in
# training, but drawn from a pair the model DID see during training (unlike
# "heldout" tasks, whose entire pair is unseen). This is the gap item 6's
# aggregation step (d) measures: heldout error vs. id_test error, both
# never trained on, differing only in whether the (drift, diffusion) PAIR
# was seen.
#
# Determinism: build_split(fraction, seed) is a pure function of
# (fraction, seed, the on-disk factorial task list) -- same seed -> same
# split, different seed -> (with very high probability) a different split.

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from data_gen.generate_factorial_benchmark import FACTORIAL_META_PARAMS_PATH
from data_gen.mechanism_library import DIFFUSION_MECHANISMS, DRIFT_MECHANISMS, MechanismTheta
from dataloaders.factorial_dataset import load_mechanism_theta_list

DRIFT_NAMES: Tuple[str, ...] = tuple(m.name for m in DRIFT_MECHANISMS)
DIFFUSION_NAMES: Tuple[str, ...] = tuple(m.name for m in DIFFUSION_MECHANISMS)
N_DRIFT = len(DRIFT_NAMES)       # 14
N_DIFFUSION = len(DIFFUSION_NAMES)  # 8
N_PAIRS = N_DRIFT * N_DIFFUSION  # 112

DEFAULT_FRACTIONS: Tuple[float, ...] = (0.10, 0.25, 0.50)
DEFAULT_ID_TEST_PER_PAIR = 2  # of 16 tasks per training pair


def n_heldout_pairs_for_fraction(fraction: float) -> int:
    """Nearest integer pair count for a holdout fraction, e.g. 0.10 -> 11,
    0.25 -> 28, 0.50 -> 56 out of 112 total pairs."""
    return round(fraction * N_PAIRS)


def _select_heldout_pairs(n_heldout: int, seed: int) -> List[Tuple[int, int]]:
    """Deterministically select `n_heldout` of the 112 (drift_idx,
    diffusion_idx) cells to hold out, subject to the coverage constraint
    (every row, every column keeps >=1 remaining training cell), balanced
    as evenly as feasible across rows and columns.

    Returns a list of (drift_idx, diffusion_idx) pairs.
    """
    if not (0 <= n_heldout <= N_PAIRS - max(N_DRIFT, N_DIFFUSION)):
        max_feasible = N_PAIRS - max(N_DRIFT, N_DIFFUSION)
        if n_heldout > max_feasible:
            raise ValueError(
                f"n_heldout={n_heldout} exceeds the max feasible under the coverage "
                f"constraint ({max_feasible} of {N_PAIRS} pairs, since each of the "
                f"{N_DRIFT} rows / {N_DIFFUSION} columns needs >=1 remaining cell)."
            )

    rng = random.Random(seed)
    cells = [(i, j) for i in range(N_DRIFT) for j in range(N_DIFFUSION)]
    rng.shuffle(cells)

    row_remaining = [N_DIFFUSION] * N_DRIFT
    col_remaining = [N_DRIFT] * N_DIFFUSION
    row_holdout = [0] * N_DRIFT
    col_holdout = [0] * N_DIFFUSION
    target_row = n_heldout / N_DRIFT
    target_col = n_heldout / N_DIFFUSION

    held: List[Tuple[int, int]] = []
    held_set = set()

    while len(held) < n_heldout:
        candidates = [
            c for c in cells
            if c not in held_set and row_remaining[c[0]] > 1 and col_remaining[c[1]] > 1
        ]
        if not candidates:
            raise RuntimeError(
                f"Ran out of feasible candidates after holding out {len(held)}/{n_heldout} "
                f"pairs under the coverage constraint -- this should not happen for "
                f"n_heldout <= {N_PAIRS - max(N_DRIFT, N_DIFFUSION)}."
            )
        # Prefer the row/column currently furthest BELOW its even-share
        # target (most under-held); candidates list order (post-shuffle)
        # breaks ties deterministically for a given seed.
        candidates.sort(key=lambda c: (row_holdout[c[0]] - target_row) + (col_holdout[c[1]] - target_col))
        pick = candidates[0]
        held.append(pick)
        held_set.add(pick)
        row_remaining[pick[0]] -= 1
        col_remaining[pick[1]] -= 1
        row_holdout[pick[0]] += 1
        col_holdout[pick[1]] += 1

    return held


def build_split(
    fraction: float,
    seed: int,
    meta_params_path: str = FACTORIAL_META_PARAMS_PATH,
    id_test_per_pair: int = DEFAULT_ID_TEST_PER_PAIR,
) -> dict:
    """Build one compositional-generalisation split manifest.

    Returns a dict (JSON-serialisable) with the held-out pair list, the
    train/id_test/heldout_test task id lists, and a coverage report.
    """
    tasks = load_mechanism_theta_list(meta_params_path)
    by_pair: Dict[Tuple[str, str], List[MechanismTheta]] = defaultdict(list)
    for t in tasks:
        by_pair[(t.drift_mechanism, t.diffusion_mechanism)].append(t)

    missing_pairs = [
        (d, s) for d in DRIFT_NAMES for s in DIFFUSION_NAMES if (d, s) not in by_pair
    ]
    if missing_pairs:
        raise RuntimeError(
            f"{len(missing_pairs)} (drift, diffusion) pairs have zero tasks in "
            f"{meta_params_path}, e.g. {missing_pairs[:3]} -- expected all {N_PAIRS} "
            f"pairs populated (14 drift x 8 diffusion mechanisms)."
        )
    pair_sizes = {k: len(v) for k, v in by_pair.items()}
    uneven = {k: n for k, n in pair_sizes.items() if n != 16}
    if uneven:
        print(f"WARNING: {len(uneven)} pairs do not have exactly 16 tasks "
              f"(expected 4 drift draws x 4 diffusion draws); e.g. {list(uneven.items())[:3]}")

    n_heldout = n_heldout_pairs_for_fraction(fraction)
    heldout_idx_pairs = _select_heldout_pairs(n_heldout, seed)
    heldout_pair_names = [(DRIFT_NAMES[i], DIFFUSION_NAMES[j]) for i, j in heldout_idx_pairs]
    heldout_pair_set = set(heldout_pair_names)

    heldout_test_ids: List[str] = []
    train_task_ids: List[str] = []
    id_test_ids: List[str] = []

    train_pair_names = [
        (d, s) for d in DRIFT_NAMES for s in DIFFUSION_NAMES if (d, s) not in heldout_pair_set
    ]

    for pair in train_pair_names:
        pair_tasks = sorted(by_pair[pair], key=lambda t: t.id)
        pair_rng = random.Random(f"{seed}|id_test|{pair[0]}|{pair[1]}")
        idx = list(range(len(pair_tasks)))
        pair_rng.shuffle(idx)
        id_test_idx = set(idx[:id_test_per_pair])
        for i, t in enumerate(pair_tasks):
            if i in id_test_idx:
                id_test_ids.append(t.id)
            else:
                train_task_ids.append(t.id)

    for pair in heldout_pair_names:
        heldout_test_ids.extend(t.id for t in by_pair[pair])

    drift_covered = {d for d, s in train_pair_names}
    diffusion_covered = {s for d, s in train_pair_names}
    coverage_ok = len(drift_covered) == N_DRIFT and len(diffusion_covered) == N_DIFFUSION

    manifest = {
        "fraction": fraction,
        "seed": seed,
        "n_drift_mechanisms": N_DRIFT,
        "n_diffusion_mechanisms": N_DIFFUSION,
        "n_pairs_total": N_PAIRS,
        "n_pairs_heldout": n_heldout,
        "n_pairs_train": len(train_pair_names),
        "id_test_per_pair": id_test_per_pair,
        "drift_mechanisms": list(DRIFT_NAMES),
        "diffusion_mechanisms": list(DIFFUSION_NAMES),
        "heldout_pairs": [list(p) for p in heldout_pair_names],
        "train_pairs": [list(p) for p in train_pair_names],
        "coverage": {
            "drift_mechanisms_covered_in_training": sorted(drift_covered),
            "diffusion_mechanisms_covered_in_training": sorted(diffusion_covered),
            "all_drift_covered": len(drift_covered) == N_DRIFT,
            "all_diffusion_covered": len(diffusion_covered) == N_DIFFUSION,
            "ok": coverage_ok,
        },
        "task_ids": {
            "train": sorted(train_task_ids),
            "id_test": sorted(id_test_ids),
            "heldout_test": sorted(heldout_test_ids),
        },
        "counts": {
            "n_train_tasks": len(train_task_ids),
            "n_id_test_tasks": len(id_test_ids),
            "n_heldout_test_tasks": len(heldout_test_ids),
        },
    }
    if not coverage_ok:
        raise RuntimeError(f"Coverage constraint violated for fraction={fraction}, seed={seed}: "
                            f"{manifest['coverage']}")
    return manifest


def print_coverage_report(manifest: dict) -> None:
    c = manifest["counts"]
    cov = manifest["coverage"]
    print(f"fraction={manifest['fraction']:.2f} seed={manifest['seed']}: "
          f"{manifest['n_pairs_heldout']}/{manifest['n_pairs_total']} pairs held out, "
          f"train_tasks={c['n_train_tasks']}, id_test_tasks={c['n_id_test_tasks']}, "
          f"heldout_test_tasks={c['n_heldout_test_tasks']}")
    print(f"  coverage: all_drift_covered={cov['all_drift_covered']} "
          f"all_diffusion_covered={cov['all_diffusion_covered']}")
    if c["n_heldout_test_tasks"] < 100:
        print(f"  WARNING: only {c['n_heldout_test_tasks']} held-out test tasks "
              f"(< 100 -- item 8 asks for 100+ held-out tasks per regime).")


def save_split(manifest: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    frac_tag = f"{round(manifest['fraction'] * 100):02d}"
    out_path = os.path.join(out_dir, f"split_frac{frac_tag}_seed{manifest['seed']}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", type=str, default=None,
                         help="Comma-separated holdout fractions, e.g. '0.10,0.25,0.50' (default).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meta-params-path", type=str, default=FACTORIAL_META_PARAMS_PATH)
    parser.add_argument("--out-dir", type=str, default="config/compositional_splits")
    parser.add_argument("--id-test-per-pair", type=int, default=DEFAULT_ID_TEST_PER_PAIR)
    args = parser.parse_args()

    fractions = (
        [float(x) for x in args.fractions.split(",")] if args.fractions else list(DEFAULT_FRACTIONS)
    )

    for fraction in fractions:
        manifest = build_split(
            fraction, args.seed, args.meta_params_path, args.id_test_per_pair,
        )
        print_coverage_report(manifest)
        out_path = save_split(manifest, args.out_dir)
        print(f"  saved -> {out_path}\n")


if __name__ == "__main__":
    main()
