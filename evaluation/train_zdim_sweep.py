"""Train models at different z_dim values for the latent dimensionality sweep.

Usage:
    PYTHONPATH=. python -u evaluation/train_zdim_sweep.py

Trains z_dim ∈ {4, 64} (z=16 already exists as meta_epoch_50.pt).
Each takes ~4-5 hrs on A10G. Saves to checkpoints/meta_zdim{N}_epoch_50.pt.
"""
import os, sys
sys.path.append(os.getcwd())

from config.base_config import cfg
from training.train_meta import train_meta_loop

ZDIMS_TO_TRAIN = [4, 64]


def train_with_zdim(z_dim):
    print(f"\n{'='*60}")
    print(f"  Training z_dim = {z_dim}")
    print(f"{'='*60}\n")

    # Override config
    cfg.latent.latent_dim = z_dim

    train_meta_loop()

    # Rename the final checkpoint
    src = "checkpoints/meta_epoch_50.pt"
    dst = f"checkpoints/meta_zdim{z_dim}_epoch_50.pt"
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"✅ Saved {dst}")
    else:
        print(f"⚠️  {src} not found after training z_dim={z_dim}")


if __name__ == '__main__':
    # Back up existing checkpoint
    existing = "checkpoints/meta_epoch_50.pt"
    backup = "checkpoints/meta_epoch_50_z16_backup.pt"
    if os.path.exists(existing) and not os.path.exists(backup):
        os.link(existing, backup)
        print(f"Backed up {existing} -> {backup}")

    for z_dim in ZDIMS_TO_TRAIN:
        train_with_zdim(z_dim)

    # Restore original z=16 checkpoint
    if os.path.exists(backup):
        if not os.path.exists(existing):
            os.rename(backup, existing)
        else:
            os.remove(backup)

    cfg.latent.latent_dim = 16
    print("\n✅ All z_dim training complete!")
