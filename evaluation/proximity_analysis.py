# evaluation/proximity_analysis.py
#
# Computes the Euclidean distance in theta-space (flattened SDE parameters)
# from each test task (regimes A, B, C) to the nearest point on the training
# manifold.  Writes results/adaptation_with_distances.csv for use by the
# plotting script.
#
# Distance metric
# ---------------
# Each SDE task theta is represented by the concatenated, flattened weight
# vectors [theta_b | theta_sigma].  Distance is measured as the L2 norm
# between this vector and the nearest training task in the same space:
#
#   d(theta_test) = min_{theta_train in Train} || v(theta_test) - v(theta_train) ||_2
#
# This gives a scalar "difficulty" score: tasks farther from the training
# manifold are structurally harder to generalise to.

import os
import sys
import torch
import numpy as np
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from config.base_config import cfg
from sde_basis.parameters import Theta  # noqa: F401 — needed for unpickling


def flatten_theta(theta: Theta) -> torch.Tensor:
    """Concatenate flattened drift and diffusion weights into one vector."""
    return torch.cat([theta.theta_b.flatten(), theta.theta_sigma.flatten()])


def main():
    print("Starting Proximity Analysis...")

    # --- 1. Load meta-parameters ---
    meta_path = cfg.paths.meta_params_path          # "data/meta_params.npz"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta-params not found at {meta_path}")

    # weights_only=False is required because meta_params contains Theta objects
    meta_params = torch.load(meta_path, map_location="cpu", weights_only=False)

    # --- 2. Build training manifold matrix ---
    train_vecs = [flatten_theta(t) for t in meta_params["train"]]
    train_matrix = torch.stack(train_vecs)  # (N_train, P)
    print(f"Training manifold: {train_matrix.shape[0]} tasks, "
          f"{train_matrix.shape[1]}-dim parameter vector")

    # --- 3. Compute per-test-task distances ---
    records = []
    for regime in ["testA", "testB", "testC"]:
        test_thetas = meta_params.get(regime, [])
        if not test_thetas:
            print(f"  WARNING: no tasks found for {regime}")
            continue

        test_vecs  = torch.stack([flatten_theta(t) for t in test_thetas])  # (N_test, P)
        dists      = torch.cdist(test_vecs, train_matrix)                  # (N_test, N_train)
        min_dists  = dists.min(dim=1).values.numpy()

        for theta, dist in zip(test_thetas, min_dists):
            records.append({
                "regime":          regime,
                "theta_id":        theta.id,
                "proximity_score": float(dist),
            })

    df_dist = pd.DataFrame(records)

    # --- 4. Merge with gated results (gate_value, mse_rollout) ---
    gated_path = "results/gated_regularized_final_fixed.csv"
    if os.path.exists(gated_path):
        df_gated = pd.read_csv(gated_path)
        # Use a single representative step count for the summary table
        canonical_steps = 50
        df_summary = df_gated[df_gated["steps_available"] == canonical_steps][
            ["theta_id", "regime", "gate_value", "mse_rollout"]
        ].copy()
        df_out = df_dist.merge(df_summary, on=["theta_id", "regime"], how="left")
    else:
        print(f"  WARNING: {gated_path} not found — saving distances only")
        df_out = df_dist

    out_path = "results/adaptation_with_distances.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved {len(df_out)} rows to {out_path}")

    # --- 5. Print per-regime statistics ---
    print("\nPer-regime summary (proximity & error at steps=50)")
    print("-" * 60)
    for regime in ["testA", "testB", "testC"]:
        sub = df_out[df_out["regime"] == regime]
        if sub.empty:
            continue
        row = (f"[{regime}]  "
               f"avg_dist={sub['proximity_score'].mean():.3f}  "
               f"max_dist={sub['proximity_score'].max():.3f}")
        if "mse_rollout" in sub.columns and sub["mse_rollout"].notna().any():
            rmse = np.sqrt(sub["mse_rollout"].dropna()).mean()
            row += f"  avg_RMSE={rmse:.4f}"
        if "gate_value" in sub.columns and sub["gate_value"].notna().any():
            row += f"  avg_gate={sub['gate_value'].mean():.4f}"
        print(row)

    # --- 6. Distance-error correlation ---
    if "mse_rollout" in df_out.columns:
        valid = df_out.dropna(subset=["mse_rollout", "proximity_score"])
        corr = valid["proximity_score"].corr(valid["mse_rollout"])
        print(f"\nCorr(distance, MSE): {corr:.4f}  "
              "(positive = farther tasks are harder)")


if __name__ == "__main__":
    main()
