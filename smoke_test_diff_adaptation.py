# smoke_test_diff_adaptation.py
#
# Item 4, step 4: smallest-possible smoke test for the differentiable
# K-step adaptation loop (adaptation/differentiable_adaptation.py,
# training/train_diff_adaptation.py).
#
# The critical risk this item calls out explicitly: an in-place optimizer
# step or a stray .detach() anywhere in the K-step loop would silently break
# the autograd graph back to the meta-parameters (the 4 global scalars for
# "global", the RateController MLP weights for "conditioned"/
# "conditioned_plus_gate", plus a freshly-initialised LearnedGateController
# for "conditioned_plus_gate") WITHOUT raising an error -- it would just
# produce a zero gradient. So this smoke test does not just check that the
# code runs; for every mode with meta-parameters, it explicitly asserts the
# post-backward() gradient norm at those meta-parameters is both finite and
# strictly positive.
#
# Also confirms, at toy scale:
#   - "fixed" constructs zero meta-parameters and still runs the full K-step
#     loop end to end with a finite query loss -- i.e. it really is just a
#     fixed-rate control, not accidentally learning anything.
#   - "global"/"conditioned"/"conditioned_plus_gate" each produce finite
#     query losses over a handful of outer meta-steps.
#
# This is NOT a real meta-training run, and NOT a comparison of the four
# variants' quality -- see training/train_diff_adaptation.py's module
# docstring for how to run that for real on GPU.
#
# Usage: python smoke_test_diff_adaptation.py

import math
import shutil
import tempfile

import torch

from config.base_config import cfg

# Toy-scale overrides, applied before any generate_*/train_meta/
# train_diff_adaptation import touches cfg (they all read cfg.* at call
# time, not import time).
cfg.basis.x_dim = 2
cfg.time_grid.n_steps = 8  # -> 9 points per trajectory, dt = T/8
cfg.dataset_sizes.n_train_thetas = 2
cfg.dataset_sizes.n_val_thetas = 1
cfg.dataset_sizes.n_train_traj_per_theta_train_inner = 4
cfg.dataset_sizes.n_train_traj_per_theta_val_inner = 2
cfg.dataset_sizes.n_val_traj_per_theta = 2
cfg.dataset_sizes.n_test_thetas = 3
cfg.dataset_sizes.n_test_support_traj_per_theta = 3
cfg.dataset_sizes.n_test_query_traj_per_theta = 3
cfg.latent.latent_dim = 4
cfg.latent.encoder_hidden_dim = 8
cfg.latent.sde_hidden_dim = 8
cfg.latent.head_hidden_dim = 8
cfg.device = "cpu"

tmp_root = tempfile.mkdtemp(prefix="smoke_test_diff_adaptation_")
cfg.paths.data_root = f"{tmp_root}/data/"
cfg.paths.meta_params_path = f"{tmp_root}/data/meta_params.npz"
cfg.paths.train_traj_root = f"{tmp_root}/data/train_trajectories/"
cfg.paths.val_traj_root = f"{tmp_root}/data/val_trajectories/"
cfg.paths.test_traj_root = f"{tmp_root}/data/test_trajectories/"

from data_gen.generate_meta_params import generate_all_meta_params
from data_gen.generate_trajectories import generate_dataset
from training.train_meta import train_meta_loop
from training.train_diff_adaptation import diff_adaptation_train_loop

# Small K -- the smoke test only needs to confirm the graph survives K
# steps intact, not to probe how K interacts with adaptation quality. Real
# runs default to DEFAULT_K=8 (see adaptation/differentiable_adaptation.py).
SMOKE_K = 3


def _run_mode(base_ckpt_path, mode, n_meta_steps=4, tasks_per_step=2):
    return diff_adaptation_train_loop(
        checkpoint_path=base_ckpt_path,
        adaptation_mode=mode,
        regime="none",
        n_meta_steps=n_meta_steps,
        tasks_per_step=tasks_per_step,
        steps_available=5,  # < n_steps+1=9 points available
        K=SMOKE_K,
        out_path=None,
        return_diagnostics=True,
    )


def main():
    print(f"Toy data root: {tmp_root}")
    generate_all_meta_params(regimes=["none"])  # train + val + testnone (support/query)
    generate_dataset()

    index_path = f"{cfg.paths.data_root}index.csv"
    ckpt_dir = f"{tmp_root}/checkpoints"
    train_meta_loop(
        loss_variant="trajectory_mse",
        n_epochs=1,
        batch_size=2,
        obs_len_range=(3, 6),
        index_path=index_path,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=1,
        max_steps_per_epoch=2,
    )
    base_ckpt_path = f"{ckpt_dir}/meta_epoch_1.pt"

    # --- (a) fixed: zero meta-parameters, pure control ---------------------
    result_fixed = _run_mode(base_ckpt_path, "fixed")
    losses_fixed = result_fixed["losses"]
    assert result_fixed["n_meta_params"] == 0, (
        f"adaptation_mode='fixed' should have zero meta-parameters, got "
        f"{result_fixed['n_meta_params']} -- it must be a pure fixed-rate "
        f"control with nothing learned."
    )
    assert len(losses_fixed) > 0, "no meta-steps ran for adaptation_mode='fixed'"
    assert all(math.isfinite(l) for l in losses_fixed), f"non-finite query loss (fixed): {losses_fixed}"
    assert all(g == 0.0 for g in result_fixed["meta_grad_norms"]), (
        "adaptation_mode='fixed' recorded a nonzero meta-gradient norm, but it "
        "has no meta-parameters to have received one from -- something is wrong."
    )
    print(f"[fixed] meta-steps ran: {len(losses_fixed)}  "
          f"losses={[f'{l:.4f}' for l in losses_fixed]}  n_meta_params=0 (confirmed no learning)")

    # --- (b)/(c)/(d): confirm nonzero, finite gradient reaches meta-params --
    for mode in ("global", "conditioned", "conditioned_plus_gate"):
        result = _run_mode(base_ckpt_path, mode)
        losses = result["losses"]
        grad_norms = result["meta_grad_norms"]

        assert result["n_meta_params"] > 0, f"adaptation_mode={mode!r} should have meta-parameters to learn"
        assert len(losses) > 0, f"no meta-steps ran for adaptation_mode={mode!r}"
        assert all(math.isfinite(l) for l in losses), f"non-finite query loss ({mode}): {losses}"

        assert len(grad_norms) > 0, f"no meta-gradients were recorded for adaptation_mode={mode!r}"
        assert all(math.isfinite(g) for g in grad_norms), (
            f"non-finite meta-parameter gradient norm ({mode}): {grad_norms}"
        )
        assert all(g > 0.0 for g in grad_norms), (
            f"adaptation_mode={mode!r} received zero gradient at its meta-parameters on some "
            f"step -- the autograd graph back through the K-step loop is broken (in-place op, "
            f"stray .detach(), or an nn.Parameter that never entered the loss): {grad_norms}"
        )
        print(f"[{mode}] meta-steps ran: {len(losses)}  n_meta_params={result['n_meta_params']}  "
              f"losses={[f'{l:.4f}' for l in losses]}  grad_norms={[f'{g:.2e}' for g in grad_norms]}")

    print("\nDifferentiable K-step adaptation smoke test passed:")
    print("  'fixed' has zero meta-parameters and changes nothing (true control).")
    print("  'global'/'conditioned'/'conditioned_plus_gate' all produce finite, strictly")
    print("  positive gradient norms at their meta-parameters after backprop through the")
    print("  K-step loop -- the autograd graph from query loss to eta/beta (and, for")
    print("  conditioned_plus_gate, to the freshly-initialised gate controller) survives")
    print("  intact. This does NOT confirm any variant is any good -- that requires a real")
    print("  GPU meta-training run (see training/train_diff_adaptation.py's module docstring).")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
