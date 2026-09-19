# data_gen/generate_factorial_benchmark.py
#
# Item 5: generator for the large factorial drift/diffusion benchmark built
# from data_gen/mechanism_library.py. This is NEW, ADDITIONAL functionality
# -- it does not touch data_gen/generate_meta_params.py or the five
# item-1 shift regimes, which items 1-4's already-queued real-GPU
# comparisons depend on. It writes to its own paths under data/ (
# factorial_meta_params.npz, factorial_trajectories/, factorial_index.csv)
# so it cannot collide with the existing train/val/test data.
#
# --- Three stages ---------------------------------------------------
#
#   params  -- build the full factorial task list (mechanism + coefficients
#              for drift x diffusion, drawn completely independently) and
#              save it. Pure CPU tensor sampling, no simulation: cheap
#              regardless of how large the factorial grid is.
#
#   verify  -- simulate real trajectories (Euler-Maruyama, via
#              simulate_batch_generic) for a moderate, category-stratified
#              subset of the full task list. Times it, measures disk usage,
#              and extrapolates to the full-scale cost -- this is item 5
#              step 6's feasibility check.
#
#   full    -- simulate trajectories for every task in the full factorial
#              list. Only run this after `verify` has shown it's
#              time/disk-feasible on the available hardware.
#
# Per-task trajectory counts and the state/time-grid conventions
# (x_dim, T, n_steps) are all read from config.base_config.cfg, matching
# what the existing test-regime pipeline (data_gen/generate_trajectories.py)
# uses -- see item 5 step 4. Trajectories are generated with the exact same
# Euler-Maruyama algorithm as data_gen/simulate_true_sde.py's simulate_batch
# (via the new simulate_batch_generic sibling function; simulate_batch
# itself is untouched), and the same retry-until-count-or-max_retries
# stability guard as generate_trajectories.py (extended here to also tally
# outcomes per drift/diffusion category -- see report_stability_stats).
#
# Usage:
#   python -m data_gen.generate_factorial_benchmark --stage params
#   python -m data_gen.generate_factorial_benchmark --stage verify
#   python -m data_gen.generate_factorial_benchmark --stage full
#
# See FACTORIAL_BENCHMARK.md for how to consume the generated benchmark.

import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from config.base_config import cfg
from sde_basis.covariance import sample_correlation
from data_gen.simulate_true_sde import simulate_batch_generic
from data_gen.mechanism_library import (
    DRIFT_MECHANISMS,
    DIFFUSION_MECHANISMS,
    MechanismTheta,
    generate_factorial_tasks,
    mechanism_drift,
    mechanism_diffusion,
)

# 14 drift mechanisms x 4 coefficient draws = 56 drift instances.
# 8 diffusion mechanisms x 4 coefficient draws = 32 diffusion instances.
# Full factorial cross: 56 x 32 = 1792 tasks. See FACTORIAL_BENCHMARK.md.
DEFAULT_N_DRIFT_PER_MECHANISM = 4
DEFAULT_N_DIFFUSION_PER_MECHANISM = 4

FACTORIAL_META_PARAMS_PATH = os.path.join(cfg.paths.data_root, "factorial_meta_params.npz")
FACTORIAL_TRAJ_ROOT = os.path.join(cfg.paths.data_root, "factorial_trajectories")
FACTORIAL_VERIFY_TRAJ_ROOT = os.path.join(cfg.paths.data_root, "factorial_trajectories_verify")


def build_tasks(
    n_drift_per_mechanism: int = DEFAULT_N_DRIFT_PER_MECHANISM,
    n_diffusion_per_mechanism: int = DEFAULT_N_DIFFUSION_PER_MECHANISM,
    seed: int = None,
) -> List[MechanismTheta]:
    gen = torch.Generator()
    gen.manual_seed(seed if seed is not None else cfg.global_seed)
    return generate_factorial_tasks(
        n_drift_per_mechanism, n_diffusion_per_mechanism, cfg.basis.x_dim, gen,
    )


def save_tasks(tasks: List[MechanismTheta], out_path: str = FACTORIAL_META_PARAMS_PATH) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"factorial": tasks}, out_path)
    print(f"Saved {len(tasks)} factorial tasks to {out_path}")


def report_category_counts(tasks: List[MechanismTheta]) -> None:
    by_drift_cat = defaultdict(set)
    by_diff_cat = defaultdict(set)
    for t in tasks:
        by_drift_cat[t.drift_category].add(t.drift_mechanism)
        by_diff_cat[t.diffusion_category].add(t.diffusion_mechanism)

    print(f"\nTotal combinations: {len(tasks)}")
    print(f"Drift mechanisms: {len(DRIFT_MECHANISMS)} across {len(by_drift_cat)} categories:")
    for cat, names in sorted(by_drift_cat.items()):
        print(f"  {cat}: {sorted(names)}")
    print(f"Diffusion mechanisms: {len(DIFFUSION_MECHANISMS)} across {len(by_diff_cat)} categories:")
    for cat, names in sorted(by_diff_cat.items()):
        print(f"  {cat}: {sorted(names)}")


def stratified_subset(tasks: List[MechanismTheta], per_pair: int = 2) -> List[MechanismTheta]:
    """Pick `per_pair` tasks from every (drift_category, diffusion_category)
    combination, so the moderate-scale verification run in step 6 spans a
    spread of categories rather than clustering on whichever mechanisms
    happen to sort first."""
    by_pair: Dict[Tuple[str, str], List[MechanismTheta]] = defaultdict(list)
    for t in tasks:
        by_pair[(t.drift_category, t.diffusion_category)].append(t)
    subset = []
    for pair in sorted(by_pair):
        subset.extend(by_pair[pair][:per_pair])
    return subset


def simulate_task(
    theta: MechanismTheta,
    out_root: str,
    roles: Sequence[Tuple[str, int]],
    generator: torch.Generator,
    device: str,
) -> Tuple[Dict[str, Tuple[int, int, int]], int]:
    """
    Simulate `roles` (role_name, count) trajectories for one MechanismTheta.

    Same retry-until-count-or-max_retries loop as
    data_gen/generate_trajectories.py, using simulate_batch_generic +
    mechanism_drift/mechanism_diffusion instead of simulate_batch + Theta.

    Returns
    -------
    role_results : role -> (n_achieved, n_requested, n_attempts)
    bytes_written : total size on disk of the .npy files written
    """
    theta_dev = theta.to(device)
    drift_fn = lambda x: mechanism_drift(x, theta_dev)
    diffusion_fn = lambda x: mechanism_diffusion(x, theta_dev)

    theta_seed = int(hash(theta.id) % 1_000_000_000)
    cov_gen = torch.Generator(device=device)
    cov_gen.manual_seed(theta_seed)
    _, L = sample_correlation(cfg.basis.x_dim, cov_gen, device)

    theta_dir = os.path.join(out_root, theta.id)
    role_results: Dict[str, Tuple[int, int, int]] = {}
    bytes_written = 0
    max_retries = cfg.stability.max_retries_per_theta

    for role, count in roles:
        if count == 0:
            continue
        valid: List[torch.Tensor] = []
        attempts = 0
        while len(valid) < count and attempts < max_retries:
            needed = count - len(valid)
            batch_size = max(min(needed * 2, 200), needed)
            x0 = torch.randn(batch_size, cfg.basis.x_dim, generator=generator, device=device)
            x0 = x0 * cfg.init.x0_std + cfg.init.x0_mean

            batch_trajs, mask = simulate_batch_generic(
                drift_fn, diffusion_fn, L, x0,
                cfg.time_grid.T, cfg.time_grid.n_steps,
                cfg.stability.max_state_abs, generator,
            )
            good = batch_trajs[mask]
            for i in range(good.shape[0]):
                valid.append(good[i].cpu())
                if len(valid) >= count:
                    break
            attempts += 1

        if valid:
            role_dir = os.path.join(theta_dir, role)
            os.makedirs(role_dir, exist_ok=True)
            for k, traj in enumerate(valid):
                fpath = os.path.join(role_dir, f"traj_{k}.npy")
                np.save(fpath, traj.numpy())
                bytes_written += os.path.getsize(fpath)

        role_results[role] = (len(valid), count, attempts)
        if len(valid) < count:
            print(f"  WARNING: {theta.id} ({role}) unstable. Got {len(valid)}/{count}")

    return role_results, bytes_written


def classify_outcome(role_results: Dict[str, Tuple[int, int, int]]) -> str:
    """clean: every role hit its full count. degraded: some role partial but
    nonzero. unusable: some role got zero usable trajectories (would be
    skipped downstream, same as the item-1 fix for ood_4/ood_5 thetas)."""
    achieved = [a for a, _, _ in role_results.values()]
    if any(a == 0 for a in achieved):
        return "unusable"
    if any(a < r for a, r, _ in role_results.values()):
        return "degraded"
    return "clean"


def simulate_tasks(
    tasks: List[MechanismTheta],
    out_root: str,
    roles: Sequence[Tuple[str, int]],
    device: str,
    sim_seed: int = None,
) -> Tuple[List[dict], float, int]:
    """Simulate `roles` trajectories for every task in `tasks`. Returns
    (index_rows, elapsed_seconds, total_bytes_written)."""
    sim_gen = torch.Generator(device=device)
    sim_gen.manual_seed(sim_seed if sim_seed is not None else cfg.global_seed + 9999)

    index_rows = []
    total_bytes = 0
    t0 = time.time()
    for i, theta in enumerate(tasks):
        role_results, bytes_written = simulate_task(theta, out_root, roles, sim_gen, device)
        total_bytes += bytes_written
        for role, (n_ok, n_req, attempts) in role_results.items():
            index_rows.append({
                "theta_id": theta.id,
                "drift_mechanism": theta.drift_mechanism,
                "drift_category": theta.drift_category,
                "diffusion_mechanism": theta.diffusion_mechanism,
                "diffusion_category": theta.diffusion_category,
                "role": role,
                "n_achieved": n_ok,
                "n_requested": n_req,
                "attempts": attempts,
            })
        if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
            print(f"  simulated {i + 1}/{len(tasks)} tasks...")
    elapsed = time.time() - t0
    return index_rows, elapsed, total_bytes


def report_stability_stats(tasks: List[MechanismTheta], index_rows: List[dict]) -> None:
    """Report clean/degraded/unusable counts by drift category and by
    diffusion category (item 5 step 5)."""
    by_theta: Dict[str, Dict[str, Tuple[int, int, int]]] = defaultdict(dict)
    for row in index_rows:
        by_theta[row["theta_id"]][row["role"]] = (row["n_achieved"], row["n_requested"], row["attempts"])

    drift_stats = defaultdict(lambda: defaultdict(int))
    diff_stats = defaultdict(lambda: defaultdict(int))
    mech_stats = defaultdict(lambda: defaultdict(int))
    for theta in tasks:
        role_results = by_theta.get(theta.id)
        if role_results is None:
            continue
        outcome = classify_outcome(role_results)
        drift_stats[theta.drift_category][outcome] += 1
        drift_stats[theta.drift_category]["total"] += 1
        diff_stats[theta.diffusion_category][outcome] += 1
        diff_stats[theta.diffusion_category]["total"] += 1
        mech_stats[theta.drift_mechanism][outcome] += 1
        mech_stats[theta.drift_mechanism]["total"] += 1

    print("\nStability by drift category (clean / degraded / unusable / total):")
    for cat, counts in sorted(drift_stats.items()):
        print(f"  {cat:20s} {counts['clean']:4d} / {counts['degraded']:4d} / "
              f"{counts['unusable']:4d} / {counts['total']:4d}")

    print("\nStability by diffusion category (clean / degraded / unusable / total):")
    for cat, counts in sorted(diff_stats.items()):
        print(f"  {cat:20s} {counts['clean']:4d} / {counts['degraded']:4d} / "
              f"{counts['unusable']:4d} / {counts['total']:4d}")

    print("\nStability by drift mechanism (clean / degraded / unusable / total):")
    for name, counts in sorted(mech_stats.items()):
        print(f"  {name:28s} {counts['clean']:4d} / {counts['degraded']:4d} / "
              f"{counts['unusable']:4d} / {counts['total']:4d}")


def _roles() -> List[Tuple[str, int]]:
    return [
        ("support", cfg.dataset_sizes.n_test_support_traj_per_theta),
        ("query", cfg.dataset_sizes.n_test_query_traj_per_theta),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["params", "verify", "full"], required=True)
    parser.add_argument("--n-drift-per-mechanism", type=int, default=DEFAULT_N_DRIFT_PER_MECHANISM)
    parser.add_argument("--n-diffusion-per-mechanism", type=int, default=DEFAULT_N_DIFFUSION_PER_MECHANISM)
    parser.add_argument("--verify-per-pair", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or "cpu"

    tasks = build_tasks(args.n_drift_per_mechanism, args.n_diffusion_per_mechanism)

    if args.stage == "params":
        save_tasks(tasks)
        report_category_counts(tasks)
        return

    roles = _roles()

    if args.stage == "verify":
        subset = stratified_subset(tasks, args.verify_per_pair)
        print(f"Verifying on {len(subset)} tasks (stratified, {args.verify_per_pair} per "
              f"drift-category x diffusion-category pair) on device={device}...")
        index_rows, elapsed, total_bytes = simulate_tasks(subset, FACTORIAL_VERIFY_TRAJ_ROOT, roles, device)
        report_stability_stats(subset, index_rows)

        n_full = len(tasks)
        per_task_time = elapsed / len(subset)
        per_task_bytes = total_bytes / len(subset)
        print(f"\nVerification run: {len(subset)} tasks, {elapsed:.1f}s total, "
              f"{per_task_time:.3f}s/task, {total_bytes / 1e6:.1f} MB total, "
              f"{per_task_bytes / 1e3:.1f} KB/task.")
        print(f"Full factorial target: {n_full} tasks.")
        print(f"Extrapolated full-scale cost: {per_task_time * n_full / 60:.1f} min, "
              f"{per_task_bytes * n_full / 1e6:.1f} MB.")
        return

    if args.stage == "full":
        print(f"Generating full factorial benchmark: {len(tasks)} tasks on device={device}...")
        save_tasks(tasks)
        index_rows, elapsed, total_bytes = simulate_tasks(tasks, FACTORIAL_TRAJ_ROOT, roles, device)
        report_stability_stats(tasks, index_rows)

        import pandas as pd
        index_path = os.path.join(cfg.paths.data_root, "factorial_index.csv")
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
        print(f"\nDone in {elapsed / 60:.1f} min, {total_bytes / 1e6:.1f} MB written. "
              f"Index saved to {index_path}.")
        return


if __name__ == "__main__":
    main()
