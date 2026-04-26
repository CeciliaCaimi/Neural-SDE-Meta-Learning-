# baselines/adapt_persistence.py
"""
Persistence Baseline
====================
Zero-parameter, zero-adaptation sanity-check lower bound.

Prediction rule: take the final observed state of the support slice
(averaged across N_SHOTS) and repeat (persist) it across every step of
the 200-step query horizon.  No model, no gradients, no checkpoint.

Because the prediction is constant in time, this baseline directly measures
how much the system drifts after the last observed moment — anything that
can't beat persistence on mse_rollout is just predicting the mean.

Outputs
-------
    results/persistence_results_full.csv
"""
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset

N_SHOTS      = 2
STEPS_SWEEP  = [20, 40, 50, 80, 100, 120, 201]
RESULTS_PATH = "results/persistence_results_full.csv"

EXPECTED_COLUMNS = [
    "regime", "theta_id", "steps_available",
    "mse_rollout", "mse_final",
    "rmse_rollout", "rmse_final",
    "mse_1step",
]


def get_task_data(dataset, theta_id, device):
    rows = dataset.metadata[dataset.metadata["theta_id"] == theta_id]
    data = [dataset[i][0] for i in rows.index.tolist()]
    return torch.stack(data).to(device)


def eval_persistence(supp_slice, query):
    """
    Args:
        supp_slice: (N_shots, steps, D) — visible portion of support
        query:      (B_q, T, D)         — full query trajectory

    Returns dict of scalar metrics (all in original simulator units).
    """
    # Average the final observed state across shots → (D,)
    last_obs = supp_slice[:, -1, :].mean(dim=0)

    B_q, T, D = query.shape
    pred = last_obs.view(1, 1, D).expand(B_q, T, D)

    mse_rollout = F.mse_loss(pred, query).item()
    mse_final   = F.mse_loss(pred[:, -1], query[:, -1]).item()
    mse_1step   = F.mse_loss(pred[:, 1],  query[:, 1]).item()

    return {
        "mse_rollout":  mse_rollout,
        "mse_final":    mse_final,
        "rmse_rollout": mse_rollout ** 0.5,
        "rmse_final":   mse_final   ** 0.5,
        "mse_1step":    mse_1step,
    }


def main():
    device = torch.device(cfg.device)
    print("⏸️  Persistence Baseline (zero-parameter sanity check)")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(RESULTS_PATH, index=False)

    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    buffer = []

    for regime in ["testA", "testB", "testC"]:
        try:
            ds_supp  = TrajectoryDataset(index_path, regime, "support")
            ds_query = TrajectoryDataset(index_path, regime, "query")
        except Exception:
            continue

        tasks = ds_supp.metadata["theta_id"].unique()

        for theta_id in tqdm(tasks, desc=regime):
            supp_full = get_task_data(ds_supp, theta_id, device)[:N_SHOTS]
            query     = get_task_data(ds_query, theta_id, device)

            for steps in STEPS_SWEEP:
                metrics = eval_persistence(supp_full[:, :steps], query)
                metrics.update({
                    "regime":          regime,
                    "theta_id":        theta_id,
                    "steps_available": steps,
                })
                buffer.append(metrics)

    df = pd.DataFrame(buffer, columns=EXPECTED_COLUMNS)
    df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)

    print(f"\n✅ Persistence results saved to {RESULTS_PATH}")
    full_df = pd.read_csv(RESULTS_PATH)
    print(full_df.groupby(["regime", "steps_available"])[
        ["mse_rollout", "mse_final", "mse_1step"]
    ].mean())


if __name__ == "__main__":
    main()
