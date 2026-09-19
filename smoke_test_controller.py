# smoke_test_controller.py
#
# Item 3, step 4: smallest-possible smoke test for the learned gate
# controller (adaptation/gate_controller.py, training/train_controller.py).
#
# Confirms, at toy scale, over a handful of tasks and controller-optimization
# steps:
#   - the training loop runs end-to-end and produces finite query losses
#   - finite, nonzero gradients actually reach the controller's parameters
#   - a_f/a_g stay within [0, 1]
#   - a_f/a_g are not degenerate (not stuck at exactly 0 or 1, and not
#     identical across every task/step from initialization)
#
# This is NOT a real meta-training run. Whether the learned controller
# actually beats the scalar gate can only be judged from a real GPU run --
# see training/train_controller.py's module docstring for how to run that
# and how to compare against the scalar gate.
#
# Usage: python smoke_test_controller.py

import math
import shutil
import tempfile

import torch

from config.base_config import cfg

# Toy-scale overrides, applied before any generate_*/train_meta/train_controller
# import touches cfg (they all read cfg.* at call time, not import time).
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

tmp_root = tempfile.mkdtemp(prefix="smoke_test_controller_")
cfg.paths.data_root = f"{tmp_root}/data/"
cfg.paths.meta_params_path = f"{tmp_root}/data/meta_params.npz"
cfg.paths.train_traj_root = f"{tmp_root}/data/train_trajectories/"
cfg.paths.val_traj_root = f"{tmp_root}/data/val_trajectories/"
cfg.paths.test_traj_root = f"{tmp_root}/data/test_trajectories/"

from data_gen.generate_meta_params import generate_all_meta_params
from data_gen.generate_trajectories import generate_dataset
from training.train_meta import train_meta_loop
from training.train_controller import controller_train_loop


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

    controller_out = f"{tmp_root}/controller.pt"
    result = controller_train_loop(
        checkpoint_path=base_ckpt_path,
        regime="none",
        n_meta_steps=4,
        tasks_per_step=2,
        steps_available=5,  # < n_steps+1=9 points available
        controller_out=controller_out,
        return_diagnostics=True,
    )

    losses = result["losses"]
    a_f_history = result["a_f_history"]
    a_g_history = result["a_g_history"]
    grad_norms = result["controller_grad_norms"]

    assert len(losses) > 0, "no controller meta-steps ran"
    assert all(math.isfinite(l) for l in losses), f"non-finite query loss: {losses}"

    assert len(grad_norms) > 0, "no controller gradients were recorded"
    assert all(math.isfinite(g) for g in grad_norms), f"non-finite controller grad norm: {grad_norms}"
    assert all(g > 0.0 for g in grad_norms), (
        f"controller received zero gradient on some step -- gradients are not "
        f"reaching the controller's parameters: {grad_norms}"
    )

    all_a = a_f_history + a_g_history
    assert len(all_a) > 0, "no (a_f, a_g) values were recorded"
    assert all(0.0 <= v <= 1.0 for v in all_a), f"a_f/a_g escaped [0, 1]: {all_a}"
    assert not all(v == 0.0 or v == 1.0 for v in all_a), (
        f"a_f/a_g are stuck at exactly 0 or 1 for every task -- degenerate gate: {all_a}"
    )
    assert len(set(round(v, 6) for v in all_a)) > 1, (
        f"a_f/a_g are identical across every task/step -- degenerate gate: {all_a}"
    )

    print("\nController smoke test passed:")
    print(f"  meta-steps ran: {len(losses)}  losses={[f'{l:.4f}' for l in losses]}")
    print(f"  a_f range=({min(a_f_history):.4f}, {max(a_f_history):.4f})  "
          f"a_g range=({min(a_g_history):.4f}, {max(a_g_history):.4f})")
    print(f"  controller grad norms={[f'{g:.2e}' for g in grad_norms]}")
    print("\nFinite non-zero gradients reach the controller; a_f/a_g stay in "
          "[0,1] and are not degenerate. This confirms the training loop is "
          "wired correctly -- it does NOT confirm the controller is any good; "
          "that requires a real GPU meta-training run (see "
          "training/train_controller.py's module docstring).")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
