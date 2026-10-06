# experiments/run_compositional.py
#
# Item 6, phase 4: the compositional-generalisation experiment driver.
#
# Three stages, selected per (method, holdout_fraction, seed):
#   train_base  -- train the arm's encoder/sde/head on the split's `train`
#                  task ids (FactorialTrajectoryDataset, role="train_inner",
#                  from data/factorial_train_index.csv -- see
#                  data_gen/generate_factorial_training_trajectories.py;
#                  NOT generated at full scale by this change, see
#                  COMPOSITIONAL_EXPERIMENT.md).
#   train_adapt -- meta-train the arm's adaptation-policy modules (only
#                  factorised_learned_rates / factorised_selective have
#                  any; a no-op for single_z / factorised), on the SAME
#                  `train` task ids, using their existing support/query
#                  trajectories (data/factorial_index.csv +
#                  factorial_trajectories/ -- these already exist at full
#                  scale, item 5).
#   evaluate    -- adapt + evaluate on the split's id_test + heldout_test
#                  task ids, writing one CSV row per task (see
#                  EVAL_COLUMNS below) to --out-csv (appended across calls,
#                  so (method, fraction, seed) sweeps accumulate into one
#                  file for experiments/aggregate_compositional.py).
#
# `seed` here is a TRAINING seed (model init, optimizer RNG, adaptation
# simulation noise) -- the compositional SPLIT (which pairs are held out)
# is controlled separately by --split-seed (default 0, the committed
# manifest under config/compositional_splits/), so every training seed for
# a given fraction is evaluated against the identical train/id_test/
# heldout_test partition. This isolates training-seed variance (item 8)
# from split variance; see COMPOSITIONAL_EXPERIMENT.md for the rationale.
#
# --dry-run prints the full (method, fraction, seed) run matrix and the
# total run count for 3 fractions x 4 methods x N seeds (default 5),
# without importing torch or touching any data.
#
# --toy builds a complete, tiny, self-contained factorial benchmark +
# compositional split under a temp directory (mutating the shared cfg
# singleton to toy sizes first, same convention as the existing
# smoke_test_*.py scripts) and runs the requested stage(s) against it --
# for CPU smoke testing only, see smoke_test_compositional.py.

import argparse
import json
import os
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

METHODS: Tuple[str, ...] = ("single_z", "factorised", "factorised_learned_rates", "factorised_selective")
DEFAULT_FRACTIONS: Tuple[float, ...] = (0.10, 0.25, 0.50)
DEFAULT_SEEDS: Tuple[int, ...] = tuple(range(5))
STAGES: Tuple[str, ...] = ("train_base", "train_adapt", "evaluate")

EVAL_COLUMNS = [
    "method", "fraction", "seed", "task_id", "drift_mechanism", "diffusion_mechanism",
    "is_heldout", "rollout_error", "drift_error", "diffusion_error", "nll",
    "a_f", "a_g", "delta_zf_norm", "delta_zg_norm",
]


def frac_tag(fraction: float) -> str:
    return f"{round(fraction * 100):02d}"


def run_tag(method: str, fraction: float, seed: int) -> str:
    return f"{method}_frac{frac_tag(fraction)}_seed{seed}"


def run_matrix(
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    methods: Sequence[str] = METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> List[Tuple[str, float, int]]:
    return [(m, f, s) for f, m, s in product(fractions, methods, seeds)]


def print_dry_run(fractions: Sequence[float], methods: Sequence[str], seeds: Sequence[int]) -> None:
    matrix = run_matrix(fractions, methods, seeds)
    print(f"Compositional experiment run matrix: {len(fractions)} fractions x "
          f"{len(methods)} methods x {len(seeds)} seeds = {len(matrix)} runs "
          f"(each run = train_base [+ train_adapt if applicable] + evaluate).")
    print(f"  fractions: {list(fractions)}")
    print(f"  methods:   {list(methods)}")
    print(f"  seeds:     {list(seeds)}")
    n_with_adapt = sum(1 for m, _, _ in matrix if m in ("factorised_learned_rates", "factorised_selective"))
    print(f"  of which {n_with_adapt}/{len(matrix)} runs also need a train_adapt stage "
          f"(single_z/factorised have no meta-parameters to meta-train).")
    print()
    for m, f, s in matrix:
        print(f"  {run_tag(m, f, s)}")


def checkpoint_paths(out_dir: str, method: str, fraction: float, seed: int) -> Dict[str, str]:
    tag = run_tag(method, fraction, seed)
    return {
        "base": os.path.join(out_dir, f"base_{tag}.pt"),
        "adapt": os.path.join(out_dir, f"adapt_{tag}.pt"),
    }


def load_split_manifest(fraction: float, split_seed: int, splits_dir: str) -> dict:
    path = os.path.join(splits_dir, f"split_frac{frac_tag(fraction)}_seed{split_seed}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split manifest not found at {path}. Generate it first with "
            f"`python -m data_gen.compositional_splits --seed {split_seed} --fractions {fraction}`."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage implementations (import torch lazily, so --dry-run never needs it).
# ---------------------------------------------------------------------------

def stage_train_base(
    method: str, fraction: float, seed: int, split_seed: int,
    out_dir: str, splits_dir: str, train_index_path: str, train_traj_root: str,
    z_dim: int, n_epochs: int, batch_size: int, obs_len_range: Tuple[int, int],
    max_steps_per_epoch: Optional[int], device: str,
) -> str:
    from config.base_config import cfg
    from dataloaders.factorial_dataset import FactorialTrajectoryDataset
    from training.train_compositional import train_base

    manifest = load_split_manifest(fraction, split_seed, splits_dir)
    train_ids = manifest["task_ids"]["train"]

    train_ds = FactorialTrajectoryDataset(train_index_path, train_traj_root, "train_inner", task_ids=train_ids)

    paths = checkpoint_paths(out_dir, method, fraction, seed)
    train_base(
        method, train_ds, cfg.basis.x_dim, z_dim, cfg,
        n_epochs=n_epochs, batch_size=batch_size, obs_len_range=obs_len_range,
        checkpoint_path=paths["base"], max_steps_per_epoch=max_steps_per_epoch,
        seed=seed, device=device,
    )
    return paths["base"]


def stage_train_adapt(
    method: str, fraction: float, seed: int, split_seed: int,
    out_dir: str, splits_dir: str, sq_index_path: str, sq_traj_root: str, meta_params_path: str,
    n_meta_steps: int, tasks_per_step: int, steps_available: int, K: int, device: str,
) -> Optional[str]:
    if method not in ("factorised_learned_rates", "factorised_selective"):
        print(f"  {method}: no meta-parameters, train_adapt is a no-op.")
        return None

    from config.base_config import cfg
    from dataloaders.factorial_dataset import FactorialTrajectoryDataset, load_mechanism_theta_dict
    from training.train_compositional import train_adapt

    manifest = load_split_manifest(fraction, split_seed, splits_dir)
    train_ids = manifest["task_ids"]["train"]
    theta_by_id = load_mechanism_theta_dict(meta_params_path)

    ds_supp = FactorialTrajectoryDataset(sq_index_path, sq_traj_root, "support", task_ids=train_ids)
    ds_query = FactorialTrajectoryDataset(sq_index_path, sq_traj_root, "query", task_ids=train_ids)

    paths = checkpoint_paths(out_dir, method, fraction, seed)
    train_adapt(
        method, paths["base"], cfg.basis.x_dim, theta_by_id, ds_supp, ds_query, train_ids, cfg,
        n_meta_steps=n_meta_steps, tasks_per_step=tasks_per_step, steps_available=steps_available,
        K=K, out_path=paths["adapt"], seed=seed, device=device,
    )
    return paths["adapt"]


def stage_evaluate(
    method: str, fraction: float, seed: int, split_seed: int,
    out_dir: str, splits_dir: str, sq_index_path: str, sq_traj_root: str, meta_params_path: str,
    out_csv: str, n_shots: int, mc_samples: int, K: int, device: str,
) -> str:
    import pandas as pd
    import torch

    from adaptation.compositional_arms import evaluate_task
    from adaptation.gated_finetuning_regularized import get_task_data
    from config.base_config import cfg
    from dataloaders.factorial_dataset import FactorialTrajectoryDataset, load_mechanism_theta_dict
    from dataloaders.trajectory_datasets import fit_scaler_on_trajectories
    from training.train_compositional import load_adapt_extra, load_base_checkpoint

    manifest = load_split_manifest(fraction, split_seed, splits_dir)
    test_ids = manifest["task_ids"]["id_test"] + manifest["task_ids"]["heldout_test"]
    heldout_set = set(manifest["task_ids"]["heldout_test"])
    theta_by_id = load_mechanism_theta_dict(meta_params_path)

    ds_supp = FactorialTrajectoryDataset(sq_index_path, sq_traj_root, "support", task_ids=test_ids)
    ds_query = FactorialTrajectoryDataset(sq_index_path, sq_traj_root, "query", task_ids=test_ids)

    device_t = torch.device(device)
    paths = checkpoint_paths(out_dir, method, fraction, seed)
    encoder, sde, head, z_dim, source_scaler = load_base_checkpoint(method, paths["base"], cfg.basis.x_dim, cfg, device_t)

    extra = {}
    if method in ("factorised_learned_rates", "factorised_selective"):
        extra = load_adapt_extra(method, paths["adapt"], z_dim, device_t)

    gen = torch.Generator(device=device_t)
    gen.manual_seed(seed + 777)

    rows = []
    for tid in test_ids:
        theta = theta_by_id[tid]
        support = get_task_data(ds_supp, tid, device_t)[:n_shots]
        query = get_task_data(ds_query, tid, device_t)
        target_scaler = fit_scaler_on_trajectories(support) if source_scaler is not None else None

        metrics = evaluate_task(
            method, encoder, sde, head, extra, support, query, theta, gen, cfg,
            target_scaler=target_scaler, mc_samples=mc_samples, K=K,
        )
        rows.append({
            "method": method, "fraction": fraction, "seed": seed, "task_id": tid,
            "drift_mechanism": theta.drift_mechanism, "diffusion_mechanism": theta.diffusion_mechanism,
            "is_heldout": tid in heldout_set,
            "rollout_error": metrics["rollout_error"], "drift_error": metrics["drift_error"],
            "diffusion_error": metrics["diffusion_error"], "nll": metrics["nll"],
            "a_f": metrics["a_f"], "a_g": metrics["a_g"],
            "delta_zf_norm": metrics["delta_zf_norm"], "delta_zg_norm": metrics["delta_zg_norm"],
        })

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(rows, columns=EVAL_COLUMNS)
    write_header = not os.path.exists(out_csv)
    df.to_csv(out_csv, mode="a" if not write_header else "w", header=write_header, index=False)
    print(f"  wrote {len(rows)} rows -> {out_csv}")
    return out_csv


# ---------------------------------------------------------------------------
# Toy environment (for --toy; also used directly by smoke_test_compositional.py).
# ---------------------------------------------------------------------------

def build_toy_environment(tmp_root: str, fractions: Sequence[float], split_seed: int = 0,
                           n_drift_per_mechanism: int = 2, n_diffusion_per_mechanism: int = 2,
                           id_test_per_pair: int = 1) -> dict:
    """Build a complete, tiny, self-contained factorial benchmark +
    compositional split(s) under `tmp_root`. Mutates the shared cfg
    singleton to toy sizes (same convention as the existing
    smoke_test_*.py scripts) -- call this BEFORE anything else touches cfg.

    Returns a dict of paths: meta_params_path, sq_index_path, sq_traj_root,
    train_index_path, train_traj_root, splits_dir.
    """
    import pandas as pd

    from config.base_config import cfg
    cfg.basis.x_dim = 2
    cfg.time_grid.n_steps = 6
    cfg.stability.max_retries_per_theta = 3
    cfg.latent.encoder_hidden_dim = 8
    cfg.latent.sde_hidden_dim = 8
    cfg.latent.head_hidden_dim = 8
    cfg.device = "cpu"

    from data_gen.compositional_splits import build_split, save_split
    from data_gen.generate_factorial_benchmark import build_tasks, save_tasks, simulate_tasks

    os.makedirs(tmp_root, exist_ok=True)
    meta_params_path = os.path.join(tmp_root, "meta_params.npz")
    tasks = build_tasks(n_drift_per_mechanism, n_diffusion_per_mechanism, seed=split_seed)
    save_tasks(tasks, meta_params_path)

    sq_traj_root = os.path.join(tmp_root, "sq_trajectories")
    train_traj_root = os.path.join(tmp_root, "train_trajectories")

    sq_rows, _, _ = simulate_tasks(tasks, sq_traj_root, [("support", 3), ("query", 3)], device="cpu")
    train_rows, _, _ = simulate_tasks(tasks, train_traj_root, [("train_inner", 4), ("val_inner", 2)], device="cpu")

    sq_index_path = os.path.join(tmp_root, "sq_index.csv")
    train_index_path = os.path.join(tmp_root, "train_index.csv")
    pd.DataFrame(sq_rows).to_csv(sq_index_path, index=False)
    pd.DataFrame(train_rows).to_csv(train_index_path, index=False)

    splits_dir = os.path.join(tmp_root, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    for fraction in fractions:
        manifest = build_split(fraction, split_seed, meta_params_path, id_test_per_pair=id_test_per_pair)
        save_split(manifest, splits_dir)

    return {
        "meta_params_path": meta_params_path,
        "sq_index_path": sq_index_path,
        "sq_traj_root": sq_traj_root,
        "train_index_path": train_index_path,
        "train_traj_root": train_traj_root,
        "splits_dir": splits_dir,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=False,
                         help="Which stage to run. Omit with --dry-run.")
    parser.add_argument("--method", type=str, choices=METHODS, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fractions", type=str, default=",".join(str(f) for f in DEFAULT_FRACTIONS))
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--toy", action="store_true", help="CPU smoke-test mode: builds its own tiny data under --toy-root.")
    parser.add_argument("--toy-root", type=str, default=None)

    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="checkpoints_compositional")
    parser.add_argument("--splits-dir", type=str, default="config/compositional_splits")
    parser.add_argument("--out-csv", type=str, default="results/compositional_results.csv")

    parser.add_argument("--z-dim", type=int, default=16, help="Per-factor latent dim for factorised arms; TOTAL latent dim for single_z (use 32 for the item-23 matched-capacity control).")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--n-meta-steps", type=int, default=200)
    parser.add_argument("--tasks-per-step", type=int, default=4)
    parser.add_argument("--steps-available", type=int, default=50)
    parser.add_argument("--k-steps", type=int, default=8)
    parser.add_argument("--n-shots", type=int, default=2)
    parser.add_argument("--mc-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    fractions = [args.fraction] if args.fraction is not None else [float(x) for x in args.fractions.split(",")]
    methods = [args.method] if args.method is not None else [m.strip() for m in args.methods.split(",")]
    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]

    if args.dry_run:
        print_dry_run(fractions, methods, seeds)
        return

    if not args.stage:
        raise SystemExit("--stage is required unless --dry-run is passed.")

    if args.toy:
        import tempfile
        toy_root = args.toy_root or tempfile.mkdtemp(prefix="run_compositional_toy_")
        paths = build_toy_environment(toy_root, fractions, split_seed=args.split_seed)
        out_dir = os.path.join(toy_root, "checkpoints")
        splits_dir = paths["splits_dir"]
        sq_index_path, sq_traj_root = paths["sq_index_path"], paths["sq_traj_root"]
        train_index_path, train_traj_root = paths["train_index_path"], paths["train_traj_root"]
        meta_params_path = paths["meta_params_path"]
        out_csv = os.path.join(toy_root, "results.csv")
        device = args.device or "cpu"
        n_epochs = min(args.n_epochs, 1)
        max_steps_per_epoch = args.max_steps_per_epoch or 2
        n_meta_steps = min(args.n_meta_steps, 2)
        tasks_per_step = min(args.tasks_per_step, 2)
        steps_available = min(args.steps_available, 3)
        k_steps = min(args.k_steps, 2)
        mc_samples = min(args.mc_samples, 2)
        z_dim = min(args.z_dim, 4)
        n_shots = min(args.n_shots, 2)
        batch_size = min(args.batch_size, 4)
        print(f"[toy mode] data under {toy_root}")
    else:
        out_dir = args.out_dir
        splits_dir = args.splits_dir
        sq_index_path = os.path.join(args.data_root, "factorial_index.csv")
        sq_traj_root = os.path.join(args.data_root, "factorial_trajectories")
        train_index_path = os.path.join(args.data_root, "factorial_train_index.csv")
        train_traj_root = os.path.join(args.data_root, "factorial_train_trajectories")
        meta_params_path = os.path.join(args.data_root, "factorial_meta_params.npz")
        out_csv = args.out_csv
        device = args.device or None
        from config.base_config import cfg as _cfg
        device = device or _cfg.device
        n_epochs, max_steps_per_epoch = args.n_epochs, args.max_steps_per_epoch
        n_meta_steps, tasks_per_step = args.n_meta_steps, args.tasks_per_step
        steps_available, k_steps, mc_samples = args.steps_available, args.k_steps, args.mc_samples
        z_dim, n_shots, batch_size = args.z_dim, args.n_shots, args.batch_size

    for method, fraction, seed in run_matrix(fractions, methods, seeds):
        print(f"=== stage={args.stage} method={method} fraction={fraction} seed={seed} ===")
        if args.stage == "train_base":
            stage_train_base(
                method, fraction, seed, args.split_seed, out_dir, splits_dir,
                train_index_path, train_traj_root, z_dim, n_epochs, batch_size,
                (20, 50) if not args.toy else (2, 5), max_steps_per_epoch, device,
            )
        elif args.stage == "train_adapt":
            stage_train_adapt(
                method, fraction, seed, args.split_seed, out_dir, splits_dir,
                sq_index_path, sq_traj_root, meta_params_path,
                n_meta_steps, tasks_per_step, steps_available, k_steps, device,
            )
        else:  # evaluate
            stage_evaluate(
                method, fraction, seed, args.split_seed, out_dir, splits_dir,
                sq_index_path, sq_traj_root, meta_params_path, out_csv,
                n_shots, mc_samples, k_steps, device,
            )


if __name__ == "__main__":
    main()
