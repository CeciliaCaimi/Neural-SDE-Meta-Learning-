# experiments/aggregate_compositional.py
#
# Item 6, phase 4: aggregation for experiments/run_compositional.py's
# per-task results CSV (EVAL_COLUMNS in that module).
#
# Produces:
#   (a) compositional_matrix(df, method, fraction, metric) -- rows=drift
#       mechanisms, cols=diffusion mechanisms, cell=mean metric over that
#       pair's tasks (averaged over seeds too), with a parallel boolean
#       matrix marking which cells were held out.
#   (b) per_fraction_summary(df, metric) -- per (method, fraction,
#       is_heldout): mean, task-level bootstrap CI, and seed-level standard
#       error (std of per-seed means / sqrt(n_seeds)).
#   (c) paired_comparison(df, method_a, method_b, metric) -- per (fraction,
#       is_heldout): mean paired difference (method_a - method_b) on tasks
#       both methods were evaluated on at the same (fraction, seed,
#       task_id), with a paired bootstrap CI.
#   (d) id_vs_heldout_gap(df, metric) -- per (method, fraction): mean
#       heldout-test error minus mean id-test error.
#
# All four work on the toy --toy CSV output (see smoke_test_compositional.py)
# as well as a real run's CSV -- nothing here assumes real-scale row counts
# beyond "at least one row per group" (empty groups are reported as NaN,
# not an error).

import argparse
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["is_heldout"] = df["is_heldout"].astype(bool)
    return df


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    """Simple task-level percentile bootstrap CI for the mean of `values`."""
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


def compositional_matrix(
    df: pd.DataFrame, method: str, fraction: float, metric: str = "rollout_error",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (value_matrix, heldout_mask) -- both (n_drift, n_diffusion)
    DataFrames indexed/columned by drift_mechanism/diffusion_mechanism,
    restricted to the tasks actually present for (method, fraction) (so a
    toy run with only a handful of mechanisms still produces a valid,
    smaller matrix instead of a mostly-NaN 14x8 grid)."""
    sub = df[(df["method"] == method) & (df["fraction"] == fraction)]
    if sub.empty:
        raise ValueError(f"No rows for method={method!r}, fraction={fraction!r}")

    value = sub.pivot_table(index="drift_mechanism", columns="diffusion_mechanism", values=metric, aggfunc="mean")
    heldout = sub.pivot_table(index="drift_mechanism", columns="diffusion_mechanism", values="is_heldout", aggfunc="mean")
    heldout = heldout.round().astype("boolean")
    return value, heldout


def format_compositional_matrix(value: pd.DataFrame, heldout: pd.DataFrame, metric: str) -> str:
    lines = [f"Compositional-generalisation matrix ({metric}, mean per (drift, diffusion) pair; "
              f"[H] = held-out pair, tasks never seen in training):"]
    col_width = max(10, max((len(c) for c in value.columns), default=10) + 4)
    header = " " * 24 + "".join(f"{c:>{col_width}}" for c in value.columns)
    lines.append(header)
    for drift in value.index:
        cells = []
        for diff in value.columns:
            v = value.loc[drift, diff]
            h = bool(heldout.loc[drift, diff]) if pd.notna(heldout.loc[drift, diff]) else False
            tag = "[H]" if h else "   "
            cell = f"{v:.4f}{tag}" if pd.notna(v) else f"{'--':>6}{tag}"
            cells.append(f"{cell:>{col_width}}")
        lines.append(f"{drift:<24}" + "".join(cells))
    return "\n".join(lines)


def per_fraction_summary(df: pd.DataFrame, metric: str = "rollout_error", n_boot: int = 1000) -> pd.DataFrame:
    """Per (method, fraction, is_heldout): n_tasks, n_seeds, mean, task-level
    bootstrap 95% CI, and seed-level standard error (std of per-seed means
    / sqrt(n_seeds)) -- item 8's "confidence intervals across tasks and
    seeds", both reported rather than only one."""
    rows = []
    for (method, fraction, is_heldout), g in df.groupby(["method", "fraction", "is_heldout"]):
        values = g[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values, n_boot=n_boot)
        per_seed_means = g.groupby("seed")[metric].mean()
        n_seeds = per_seed_means.shape[0]
        seed_se = float(per_seed_means.std(ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else float("nan")
        rows.append({
            "method": method, "fraction": fraction, "is_heldout": is_heldout,
            "n_tasks": len(values), "n_seeds": n_seeds,
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "seed_se": seed_se,
        })
    return pd.DataFrame(rows).sort_values(["fraction", "method", "is_heldout"]).reset_index(drop=True)


def paired_comparison(
    df: pd.DataFrame, method_a: str, method_b: str, metric: str = "rollout_error", n_boot: int = 1000,
) -> pd.DataFrame:
    """Per (fraction, is_heldout): mean(method_a - method_b) on tasks both
    methods share at the same (fraction, seed, task_id), with a paired
    bootstrap CI on the per-task difference."""
    key = ["fraction", "seed", "task_id", "is_heldout"]
    a = df[df["method"] == method_a][key + [metric]].rename(columns={metric: "a"})
    b = df[df["method"] == method_b][key + [metric]].rename(columns={metric: "b"})
    merged = a.merge(b, on=key, how="inner")
    diff_col = f"mean_diff_({method_a}-{method_b})"
    empty_cols = ["fraction", "is_heldout", "n_paired_tasks", diff_col, "ci_lo", "ci_hi"]
    if merged.empty:
        return pd.DataFrame(columns=empty_cols)
    merged["diff"] = merged["a"] - merged["b"]

    rows = []
    for (fraction, is_heldout), g in merged.groupby(["fraction", "is_heldout"]):
        diffs = g["diff"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(diffs, n_boot=n_boot)
        rows.append({
            "fraction": fraction, "is_heldout": is_heldout, "n_paired_tasks": len(diffs),
            diff_col: float(np.mean(diffs)) if len(diffs) else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values(["fraction", "is_heldout"]).reset_index(drop=True)


def id_vs_heldout_gap(df: pd.DataFrame, metric: str = "rollout_error") -> pd.DataFrame:
    """Per (method, fraction): mean heldout-test error minus mean id-test
    error (positive -> compositional generalisation is harder than
    in-distribution, as expected)."""
    rows = []
    for (method, fraction), g in df.groupby(["method", "fraction"]):
        id_mean = g.loc[~g["is_heldout"], metric].mean()
        heldout_mean = g.loc[g["is_heldout"], metric].mean()
        rows.append({
            "method": method, "fraction": fraction,
            "id_test_mean": id_mean, "heldout_test_mean": heldout_mean,
            "gap_heldout_minus_id": heldout_mean - id_mean,
        })
    return pd.DataFrame(rows).sort_values(["fraction", "method"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=str, default="results/compositional_results.csv")
    parser.add_argument("--metric", type=str, default="rollout_error")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--compare", type=str, default=None,
                         help="Comma-separated pair of methods for a paired comparison, e.g. 'factorised,single_z'.")
    args = parser.parse_args()

    df = load_results(args.results_csv)
    methods = sorted(df["method"].unique())
    fractions = sorted(df["fraction"].unique())

    print("=" * 70)
    print("(a) Compositional-generalisation matrices")
    print("=" * 70)
    for method in methods:
        for fraction in fractions:
            sub = df[(df["method"] == method) & (df["fraction"] == fraction)]
            if sub.empty:
                continue
            value, heldout = compositional_matrix(df, method, fraction, args.metric)
            print(f"\nmethod={method} fraction={fraction}")
            print(format_compositional_matrix(value, heldout, args.metric))

    print("\n" + "=" * 70)
    print("(b) Per-fraction summaries (mean, task-level bootstrap CI, seed-level SE)")
    print("=" * 70)
    summary = per_fraction_summary(df, args.metric, args.n_boot)
    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("(c) Paired comparisons between methods on identical tasks")
    print("=" * 70)
    if args.compare:
        ma, mb = [m.strip() for m in args.compare.split(",")]
        pairs = [(ma, mb)]
    else:
        pairs = [(methods[i], methods[j]) for i in range(len(methods)) for j in range(i + 1, len(methods))]
    for ma, mb in pairs:
        pc = paired_comparison(df, ma, mb, args.metric, args.n_boot)
        if pc.empty:
            continue
        print(f"\n{ma} vs {mb}:")
        print(pc.to_string(index=False))

    print("\n" + "=" * 70)
    print("(d) ID-test vs held-out-test gap")
    print("=" * 70)
    gap = id_vs_heldout_gap(df, args.metric)
    print(gap.to_string(index=False))


if __name__ == "__main__":
    main()
