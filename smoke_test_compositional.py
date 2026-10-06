# smoke_test_compositional.py
#
# Item 6, phase 5: smallest-possible smoke test for the compositional-
# generalisation experiment (data_gen/compositional_splits.py,
# adaptation/compositional_arms.py, models/single_latent_encoder.py,
# training/train_compositional.py, experiments/run_compositional.py,
# experiments/aggregate_compositional.py).
#
# Confirms, at toy scale:
#   - split counts are exactly 11/28/56 held-out pairs for fractions
#     0.10/0.25/0.50 against the REAL (already-generated) 1792-task
#     factorial benchmark, with the row/column coverage constraint holding
#     and >=100 held-out test tasks for every fraction.
#   - splits are deterministic given (fraction, seed) and sensitive to seed
#     changes; train/id_test/heldout_test never overlap and cover every task.
#   - each of the four arms (single_z, factorised, factorised_learned_rates,
#     factorised_selective) runs train_base -> train_adapt -> evaluate
#     end to end on a tiny synthetic factorial benchmark, producing finite
#     metrics.
#   - single_z really ties the latent: delta_zf_norm == delta_zg_norm
#     exactly (one leaf tensor, not two coincidentally-equal ones).
#   - experiments/aggregate_compositional.py's four aggregation functions
#     run on the toy results CSV and produce a non-empty compositional
#     matrix.
#
# This is NOT a real experiment -- toy x_dim, a couple of mechanism draws,
# a handful of optimizer/meta-steps. See COMPOSITIONAL_EXPERIMENT.md for
# the real run.
#
# Usage: python smoke_test_compositional.py

import math
import os
import shutil
import tempfile

import torch

from data_gen.generate_factorial_benchmark import FACTORIAL_META_PARAMS_PATH


def check_real_split_counts():
    from data_gen.compositional_splits import build_split, n_heldout_pairs_for_fraction

    expected = {0.10: 11, 0.25: 28, 0.50: 56}
    manifests = {}
    for fraction, expected_n in expected.items():
        n = n_heldout_pairs_for_fraction(fraction)
        assert n == expected_n, f"fraction={fraction}: expected {expected_n} held-out pairs, got {n}"
        manifest = build_split(fraction, seed=0, meta_params_path=FACTORIAL_META_PARAMS_PATH)
        assert manifest["n_pairs_heldout"] == expected_n
        assert manifest["coverage"]["ok"], f"coverage constraint violated for fraction={fraction}: {manifest['coverage']}"
        n_heldout_tasks = manifest["counts"]["n_heldout_test_tasks"]
        assert n_heldout_tasks >= 100, f"fraction={fraction}: only {n_heldout_tasks} held-out test tasks (<100)"
        manifests[fraction] = manifest

    # Determinism + seed sensitivity + no leakage, on fraction=0.25.
    m_a = build_split(0.25, seed=0, meta_params_path=FACTORIAL_META_PARAMS_PATH)
    m_b = build_split(0.25, seed=0, meta_params_path=FACTORIAL_META_PARAMS_PATH)
    m_c = build_split(0.25, seed=1, meta_params_path=FACTORIAL_META_PARAMS_PATH)
    assert m_a["task_ids"] == m_b["task_ids"] and m_a["heldout_pairs"] == m_b["heldout_pairs"], (
        "same (fraction, seed) produced different splits -- not deterministic"
    )
    assert m_a["task_ids"] != m_c["task_ids"] or m_a["heldout_pairs"] != m_c["heldout_pairs"], (
        "different seeds produced an identical split"
    )

    train = set(m_a["task_ids"]["train"])
    id_test = set(m_a["task_ids"]["id_test"])
    heldout = set(m_a["task_ids"]["heldout_test"])
    assert not (train & id_test), "train/id_test overlap"
    assert not (train & heldout), "train/heldout overlap"
    assert not (id_test & heldout), "id_test/heldout overlap"
    assert len(train) + len(id_test) + len(heldout) == 1792, "split does not cover all 1792 tasks"

    print("Split checks passed: 11/28/56 held-out pairs for 0.10/0.25/0.50, coverage constraint holds, "
          ">=100 held-out test tasks for every fraction, deterministic given seed, seed-sensitive, no leakage.")


def check_four_arms_end_to_end():
    from adaptation.compositional_arms import ARMS, encode
    from experiments.run_compositional import build_toy_environment, checkpoint_paths
    from training.train_compositional import load_adapt_extra, load_base_checkpoint, train_adapt, train_base
    from dataloaders.factorial_dataset import FactorialTrajectoryDataset, load_mechanism_theta_dict
    from adaptation.gated_finetuning_regularized import get_task_data
    from dataloaders.trajectory_datasets import fit_scaler_on_trajectories
    from adaptation.compositional_arms import evaluate_task
    from config.base_config import cfg

    tmp_root = tempfile.mkdtemp(prefix="smoke_test_compositional_")
    try:
        fraction = 0.5
        paths = build_toy_environment(tmp_root, fractions=[fraction], split_seed=0,
                                       n_drift_per_mechanism=2, n_diffusion_per_mechanism=2,
                                       id_test_per_pair=1)

        import json
        with open(os.path.join(paths["splits_dir"], "split_frac50_seed0.json")) as f:
            manifest = json.load(f)
        train_ids = manifest["task_ids"]["train"]
        test_ids = manifest["task_ids"]["id_test"] + manifest["task_ids"]["heldout_test"]
        assert len(train_ids) > 0 and len(test_ids) > 0

        theta_by_id = load_mechanism_theta_dict(paths["meta_params_path"])
        out_dir = os.path.join(tmp_root, "checkpoints")
        device = torch.device("cpu")

        for arm in ARMS:
            print(f"\n--- arm={arm} ---")
            train_ds = FactorialTrajectoryDataset(
                paths["train_index_path"], paths["train_traj_root"], "train_inner", task_ids=train_ids,
            )
            ckpt_paths = checkpoint_paths(out_dir, arm, fraction, seed=0)
            losses = train_base(
                arm, train_ds, x_dim=cfg.basis.x_dim, z_dim=4, cfg=cfg,
                n_epochs=1, batch_size=4, obs_len_range=(2, 5),
                checkpoint_path=ckpt_paths["base"], max_steps_per_epoch=2, seed=0,
                return_losses=True,
            )
            assert len(losses) > 0, f"{arm}: no train_base steps ran"
            assert all(math.isfinite(l) for l in losses), f"{arm}: non-finite train_base loss {losses}"

            ds_supp = FactorialTrajectoryDataset(paths["sq_index_path"], paths["sq_traj_root"], "support", task_ids=train_ids + test_ids)
            ds_query = FactorialTrajectoryDataset(paths["sq_index_path"], paths["sq_traj_root"], "query", task_ids=train_ids + test_ids)

            if arm in ("factorised_learned_rates", "factorised_selective"):
                diag = train_adapt(
                    arm, ckpt_paths["base"], cfg.basis.x_dim, theta_by_id, ds_supp, ds_query, train_ids, cfg,
                    n_meta_steps=2, tasks_per_step=2, steps_available=3, K=2,
                    out_path=ckpt_paths["adapt"], seed=0, return_diagnostics=True,
                )
                assert len(diag["losses"]) > 0, f"{arm}: no train_adapt meta-steps ran"
                assert all(math.isfinite(l) for l in diag["losses"]), f"{arm}: non-finite train_adapt loss"
                assert all(g > 0.0 for g in diag["grad_norms"]), (
                    f"{arm}: zero gradient reached the adaptation-policy meta-parameters"
                )

            encoder, sde, head, z_dim, source_scaler = load_base_checkpoint(arm, ckpt_paths["base"], cfg.basis.x_dim, cfg, device)
            extra = {}
            if arm in ("factorised_learned_rates", "factorised_selective"):
                extra = load_adapt_extra(arm, ckpt_paths["adapt"], z_dim, device)

            gen = torch.Generator(device=device)
            gen.manual_seed(0)
            for tid in test_ids[:2]:
                theta = theta_by_id[tid]
                support = get_task_data(ds_supp, tid, device)[:2]
                query = get_task_data(ds_query, tid, device)
                target_scaler = fit_scaler_on_trajectories(support) if source_scaler is not None else None
                metrics = evaluate_task(
                    arm, encoder, sde, head, extra, support, query, theta, gen, cfg,
                    target_scaler=target_scaler, mc_samples=2, K=2,
                )
                for key in ("rollout_error", "drift_error", "diffusion_error", "nll",
                            "delta_zf_norm", "delta_zg_norm"):
                    assert math.isfinite(metrics[key]), f"{arm}/{tid}: non-finite {key}={metrics[key]}"

                if arm == "single_z":
                    assert metrics["delta_zf_norm"] == metrics["delta_zg_norm"], (
                        f"single_z must tie delta_zf_norm == delta_zg_norm exactly, got "
                        f"{metrics['delta_zf_norm']} != {metrics['delta_zg_norm']}"
                    )
                if arm == "factorised_selective":
                    assert 0.0 <= metrics["a_f"] <= 1.0 and 0.0 <= metrics["a_g"] <= 1.0, (
                        f"factorised_selective: a_f/a_g escaped [0,1]: {metrics['a_f']}, {metrics['a_g']}"
                    )

            print(f"  {arm}: train_base ({len(losses)} steps) + "
                  f"{'train_adapt + ' if arm in ('factorised_learned_rates', 'factorised_selective') else ''}"
                  f"evaluate ({len(test_ids[:2])} tasks) all produced finite metrics.")

        # Explicit tying check at the encode() level too, not just via evaluate_task.
        from models.single_latent_encoder import SingleLatentEncoder
        enc = SingleLatentEncoder(x_dim=2, z_dim=4, hidden_dim=8)
        obs = torch.randn(3, 5, 2)
        zf, zg = encode("single_z", enc, obs)
        assert zf is zg, "single_z's encode() must return the SAME tensor object for z_f and z_g"

        print("\nAll four arms ran end-to-end (train_base [+ train_adapt] + evaluate) with finite "
              "metrics; single_z's latent is structurally tied (same tensor object, not merely "
              "equal values).")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def check_aggregation():
    import pandas as pd

    from experiments.aggregate_compositional import (
        compositional_matrix, id_vs_heldout_gap, load_results, paired_comparison, per_fraction_summary,
    )

    tmp_root = tempfile.mkdtemp(prefix="smoke_test_compositional_agg_")
    try:
        rows = []
        for method in ("factorised", "single_z"):
            for seed in (0, 1):
                for i, (drift, diff_mech, is_h) in enumerate([
                    ("ou_linear", "const_diffusion", False),
                    ("ou_linear", "cir_sqrt", True),
                    ("double_well", "const_diffusion", True),
                    ("double_well", "cir_sqrt", False),
                ]):
                    rows.append({
                        "method": method, "fraction": 0.25, "seed": seed, "task_id": f"task_{seed}_{i}",
                        "drift_mechanism": drift, "diffusion_mechanism": diff_mech, "is_heldout": is_h,
                        "rollout_error": 0.1 + 0.01 * i + (0.02 if method == "single_z" else 0.0),
                        "drift_error": 0.05, "diffusion_error": 0.02, "nll": 10.0,
                        "a_f": float("nan"), "a_g": float("nan"),
                        "delta_zf_norm": 0.3, "delta_zg_norm": 0.3,
                    })
        csv_path = os.path.join(tmp_root, "toy_results.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        df = load_results(csv_path)
        value, heldout = compositional_matrix(df, "factorised", 0.25, "rollout_error")
        assert value.shape[0] > 0 and value.shape[1] > 0, "compositional matrix is empty"
        assert heldout.to_numpy().any(), "expected at least one held-out cell marked"

        summary = per_fraction_summary(df, "rollout_error", n_boot=50)
        assert len(summary) > 0, "per_fraction_summary produced no rows"
        assert summary["mean"].notna().all(), "per_fraction_summary has NaN means"

        pc = paired_comparison(df, "factorised", "single_z", "rollout_error", n_boot=50)
        assert len(pc) > 0, "paired_comparison produced no rows (methods/tasks didn't align)"

        gap = id_vs_heldout_gap(df, "rollout_error")
        assert len(gap) > 0, "id_vs_heldout_gap produced no rows"

        print("Aggregation checks passed: compositional matrix, per-fraction summary (with CIs), "
              "paired comparison, and ID-vs-heldout gap all ran on a toy results CSV.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    check_real_split_counts()
    check_four_arms_end_to_end()
    check_aggregation()
    print("\nCompositional-generalisation smoke test passed. This confirms the splits, the four "
          "arms' train_base/train_adapt/evaluate wiring, and the aggregation functions are correct "
          "at toy scale -- it does NOT confirm any method is better than another; see "
          "COMPOSITIONAL_EXPERIMENT.md for the real run.")


if __name__ == "__main__":
    main()
