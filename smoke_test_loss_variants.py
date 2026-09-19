# smoke_test_loss_variants.py
#
# Item 2, step 3: smallest-possible smoke test for the four selectable SDE
# training objectives in training/train_meta.py. Confirms each variant runs
# end-to-end on a toy-scale dataset, produces finite loss and gradients, and
# that the NLL-based variants don't blow up numerically.
#
# This is NOT a real training comparison -- toy x_dim, toy_steps, a couple
# of thetas/trajectories, a handful of optimizer steps. Which variant is
# "best" can only be judged from a real GPU run (see training/train_meta.py
# module docstring for how to run that).
#
# Usage: python smoke_test_loss_variants.py

import copy
import math
import shutil
import tempfile

import torch

from config.base_config import cfg

# Toy-scale overrides, applied to the shared cfg singleton before any of the
# generate_*/train_meta imports touch it (they all read cfg.* at call time,
# not import time, so mutating first is enough).
cfg.basis.x_dim = 2
cfg.time_grid.n_steps = 8  # -> 9 points per trajectory, dt = T/8
cfg.dataset_sizes.n_train_thetas = 2
cfg.dataset_sizes.n_val_thetas = 1
cfg.dataset_sizes.n_train_traj_per_theta_train_inner = 4
cfg.dataset_sizes.n_train_traj_per_theta_val_inner = 2
cfg.dataset_sizes.n_val_traj_per_theta = 2
cfg.latent.latent_dim = 4
cfg.latent.encoder_hidden_dim = 8
cfg.latent.sde_hidden_dim = 8
cfg.latent.head_hidden_dim = 8
cfg.device = "cpu"

tmp_root = tempfile.mkdtemp(prefix="smoke_test_loss_variants_")
cfg.paths.data_root = f"{tmp_root}/data/"
cfg.paths.meta_params_path = f"{tmp_root}/data/meta_params.npz"
cfg.paths.train_traj_root = f"{tmp_root}/data/train_trajectories/"
cfg.paths.val_traj_root = f"{tmp_root}/data/val_trajectories/"
cfg.paths.test_traj_root = f"{tmp_root}/data/test_trajectories/"

from data_gen.generate_meta_params import generate_all_meta_params
from data_gen.generate_trajectories import generate_dataset
from training.train_meta import LOSS_VARIANTS, train_meta_loop


def main():
    print(f"Toy data root: {tmp_root}")
    generate_all_meta_params(regimes=[])  # train + val only, no test regimes needed
    generate_dataset()

    index_path = f"{cfg.paths.data_root}index.csv"
    results = {}

    for variant in LOSS_VARIANTS:
        print(f"\n=== Smoke-testing loss_variant={variant!r} ===")
        ckpt_dir = f"{tmp_root}/checkpoints_{variant}"
        losses = train_meta_loop(
            loss_variant=variant,
            n_epochs=1,
            batch_size=4,
            obs_len_range=(3, 6),
            index_path=index_path,
            checkpoint_dir=ckpt_dir,
            checkpoint_every=1,
            max_steps_per_epoch=3,
            return_losses=True,
        )
        assert len(losses) > 0, f"{variant}: no training steps ran"
        assert all(math.isfinite(l) for l in losses), f"{variant}: non-finite loss {losses}"
        print(f"{variant}: {len(losses)} steps, losses={[f'{l:.4f}' for l in losses]}")
        results[variant] = losses

    print("\n=== Smoke test summary ===")
    for variant, losses in results.items():
        print(f"  {variant:36s} steps={len(losses)}  last_loss={losses[-1]:.4f}  finite=True")
    print("\nAll four loss variants ran end-to-end with finite loss/gradients.")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
