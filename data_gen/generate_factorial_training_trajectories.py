# data_gen/generate_factorial_training_trajectories.py
#
# Item 6, phase 1c.
#
# Phase 0 found a mismatch: data/factorial_trajectories/ (item 5) only has
# `support` (10) and `query` (32) trajectories per task -- the right shape
# for test-time few-shot adaptation + evaluation, but NOT what
# training/train_meta.py's meta-training loop needs. That loop trains the
# encoder/sde/head themselves by teacher-forcing on many (~80) `train_inner`
# trajectories per training theta (plus ~20 `val_inner` for monitoring),
# batched and shuffled across many thetas -- a different role, and a much
# larger per-task trajectory count, than `support`/`query`.
#
# This script adds exactly those missing `train_inner`/`val_inner`
# trajectories for a given explicit list of factorial task ids (item 6
# phase 1d: callers pass task ids, not a regime name), writing them to a
# NEW directory (default data/factorial_train_trajectories/) and a NEW
# index (default data/factorial_train_index.csv) -- it never writes into
# data/factorial_trajectories/ or data/factorial_index.csv, so the existing,
# already-verified item-5 benchmark (FACTORIAL_BENCHMARK.md) is untouched.
#
# Simulation itself reuses data_gen/generate_factorial_benchmark.py's
# simulate_tasks/simulate_task/report_stability_stats as-is (imported, not
# duplicated) -- those functions are already generic over (tasks, out_root,
# roles, device), so no new simulation code is needed here.
#
# --- Stages ---------------------------------------------------------------
#   verify -- simulate train_inner/val_inner for a SMALL subset of the
#             requested task ids (--verify-n-tasks, default 3), time it,
#             check free disk space, and extrapolate the full-scale
#             time/disk cost for the FULL requested task-id list. Always
#             safe to run (toy by construction).
#   full   -- simulate train_inner/val_inner for every requested task id.
#             NOT run as part of this change -- see
#             COMPOSITIONAL_EXPERIMENT.md for why, and run this manually
#             (after reviewing the `verify` stage's time/disk estimate) when
#             ready for the real GPU/CPU training run.
#
# Usage:
#   python -m data_gen.generate_factorial_training_trajectories \
#       --stage verify --task-ids-file <path, one theta_id per line>
#   python -m data_gen.generate_factorial_training_trajectories \
#       --stage full --task-ids-file <path> --confirm-full-run

import argparse
import os
import shutil
import time
from typing import List, Optional

import pandas as pd
import torch

from config.base_config import cfg
from data_gen.generate_factorial_benchmark import FACTORIAL_META_PARAMS_PATH, report_stability_stats, simulate_tasks
from data_gen.mechanism_library import MechanismTheta
from dataloaders.factorial_dataset import load_mechanism_theta_dict

DEFAULT_N_TRAIN_INNER = 80
DEFAULT_N_VAL_INNER = 20

FACTORIAL_TRAIN_TRAJ_ROOT = os.path.join(cfg.paths.data_root, "factorial_train_trajectories")
FACTORIAL_TRAIN_INDEX_PATH = os.path.join(cfg.paths.data_root, "factorial_train_index.csv")


def load_task_ids(task_ids_file: str) -> List[str]:
    with open(task_ids_file) as f:
        return [line.strip() for line in f if line.strip()]


def tasks_for_ids(meta_params_path: str, task_ids: List[str]) -> List[MechanismTheta]:
    by_id = load_mechanism_theta_dict(meta_params_path)
    missing = [tid for tid in task_ids if tid not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} task id(s) not found in {meta_params_path}, e.g. {missing[:5]}")
    return [by_id[tid] for tid in task_ids]


def check_disk_space(estimated_bytes: float, out_root: str, safety_factor: float = 1.5) -> None:
    """Raise if the estimated write would not comfortably fit in free disk
    space under out_root's filesystem. Checked BEFORE any writing happens
    (parent dir need not exist yet -- walk up until an existing ancestor is
    found)."""
    probe = os.path.abspath(out_root)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    free = shutil.disk_usage(probe).free
    needed = estimated_bytes * safety_factor
    print(f"Disk check: estimated {estimated_bytes / 1e6:.1f} MB "
          f"(x{safety_factor} safety factor = {needed / 1e6:.1f} MB needed), "
          f"{free / 1e6:.1f} MB free at {probe}.")
    if needed > free:
        raise RuntimeError(
            f"Not enough free disk space: need ~{needed / 1e6:.1f} MB "
            f"(with safety factor), only {free / 1e6:.1f} MB free at {probe}. Aborting before writing anything."
        )


def _roles(n_train_inner: int, n_val_inner: int):
    return [("train_inner", n_train_inner), ("val_inner", n_val_inner)]


def run_verify(
    task_ids: List[str],
    meta_params_path: str,
    out_root: str,
    index_out: str,
    n_train_inner: int,
    n_val_inner: int,
    verify_n_tasks: int,
    device: str,
) -> None:
    all_tasks = tasks_for_ids(meta_params_path, task_ids)
    subset = all_tasks[:verify_n_tasks]
    print(f"Verifying train_inner/val_inner generation on {len(subset)}/{len(all_tasks)} "
          f"requested tasks (roles=train_inner:{n_train_inner}, val_inner:{n_val_inner}), device={device}...")

    verify_root = out_root.rstrip("/") + "_verify"
    roles = _roles(n_train_inner, n_val_inner)

    # Rough pre-check using item-5's measured 343 KB/task for 10+32=42
    # trajectories -> ~8.17 KB/trajectory -> scale to the requested counts.
    bytes_per_traj_estimate = 343_000.0 / 42.0
    rough_estimate = bytes_per_traj_estimate * (n_train_inner + n_val_inner) * len(all_tasks)
    check_disk_space(rough_estimate, verify_root)

    index_rows, elapsed, total_bytes = simulate_tasks(subset, verify_root, roles, device)
    report_stability_stats(subset, index_rows)

    per_task_time = elapsed / len(subset)
    per_task_bytes = total_bytes / len(subset)
    n_full = len(all_tasks)
    print(f"\nVerification run: {len(subset)} tasks, {elapsed:.1f}s total, "
          f"{per_task_time:.3f}s/task, {total_bytes / 1e6:.2f} MB total, "
          f"{per_task_bytes / 1e3:.1f} KB/task.")
    print(f"Full requested-task-list target: {n_full} tasks.")
    print(f"Extrapolated full-scale cost: {per_task_time * n_full / 60:.1f} min, "
          f"{per_task_bytes * n_full / 1e6:.1f} MB.")
    print(f"(This writes to {verify_root}, NOT {out_root} -- the 'full' stage, "
          f"not run here, is what would write trajectories under {out_root} "
          f"and an index to {index_out}.)")

    # Re-check disk space with the measured per-task rate for a tighter bound.
    check_disk_space(per_task_bytes * n_full, out_root)

    shutil.rmtree(verify_root, ignore_errors=True)


def run_full(
    task_ids: List[str],
    meta_params_path: str,
    out_root: str,
    index_out: str,
    n_train_inner: int,
    n_val_inner: int,
    device: str,
) -> None:
    tasks = tasks_for_ids(meta_params_path, task_ids)
    roles = _roles(n_train_inner, n_val_inner)

    bytes_per_traj_estimate = 343_000.0 / 42.0
    rough_estimate = bytes_per_traj_estimate * (n_train_inner + n_val_inner) * len(tasks)
    check_disk_space(rough_estimate, out_root)

    print(f"Generating train_inner/val_inner for {len(tasks)} tasks on device={device}...")
    index_rows, elapsed, total_bytes = simulate_tasks(tasks, out_root, roles, device)
    report_stability_stats(tasks, index_rows)

    os.makedirs(os.path.dirname(index_out) or ".", exist_ok=True)
    pd.DataFrame(index_rows).to_csv(index_out, index=False)
    print(f"\nDone in {elapsed / 60:.1f} min, {total_bytes / 1e6:.1f} MB written. "
          f"Index saved to {index_out}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["verify", "full"], required=True)
    parser.add_argument("--task-ids-file", type=str, required=True,
                         help="Text file, one factorial theta_id per line.")
    parser.add_argument("--meta-params-path", type=str, default=FACTORIAL_META_PARAMS_PATH)
    parser.add_argument("--out-root", type=str, default=FACTORIAL_TRAIN_TRAJ_ROOT)
    parser.add_argument("--index-out", type=str, default=FACTORIAL_TRAIN_INDEX_PATH)
    parser.add_argument("--n-train-inner", type=int, default=DEFAULT_N_TRAIN_INNER)
    parser.add_argument("--n-val-inner", type=int, default=DEFAULT_N_VAL_INNER)
    parser.add_argument("--verify-n-tasks", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--confirm-full-run", action="store_true",
                         help="Required for --stage full, as an explicit guard against "
                              "accidentally triggering a large run.")
    args = parser.parse_args()

    task_ids = load_task_ids(args.task_ids_file)
    if not task_ids:
        raise ValueError(f"No task ids found in {args.task_ids_file}")

    if args.stage == "verify":
        run_verify(
            task_ids, args.meta_params_path, args.out_root, args.index_out,
            args.n_train_inner, args.n_val_inner, args.verify_n_tasks, args.device,
        )
    else:
        if not args.confirm_full_run:
            raise SystemExit(
                "--stage full requires --confirm-full-run (safety guard). "
                "Run --stage verify first and review the time/disk estimate."
            )
        run_full(
            task_ids, args.meta_params_path, args.out_root, args.index_out,
            args.n_train_inner, args.n_val_inner, args.device,
        )


if __name__ == "__main__":
    main()
