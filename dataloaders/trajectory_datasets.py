# dataloaders/trajectory_datasets.py

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

# We import cfg to sanity-check shapes against the global configuration
from config.base_config import cfg


# ---------------------------------------------------------------------------
# Module-level scaler helpers (two-scalar approach)
# ---------------------------------------------------------------------------

def fit_scaler_on_trajectories(trajectories: torch.Tensor) -> StandardScaler:
    """Fit a per-dimension StandardScaler on a batch of trajectories.

    Args:
        trajectories: (N, T, D) — N trajectories, T time-steps, D state dims.
    Returns:
        Fitted StandardScaler whose statistics span all N*T observations.
    """
    N, T, D = trajectories.shape
    flat = trajectories.reshape(-1, D).cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def apply_scaler_to_trajectories(
    trajectories: torch.Tensor, scaler: StandardScaler
) -> torch.Tensor:
    """Apply a fitted scaler to trajectories, preserving device and dtype.

    Args:
        trajectories: (N, T, D).
        scaler: a fitted StandardScaler.
    Returns:
        Normalized (N, T, D) tensor on the original device.
    """
    device = trajectories.device
    N, T, D = trajectories.shape
    flat = trajectories.reshape(-1, D).cpu().numpy()
    normalized = scaler.transform(flat).astype(np.float32)
    return torch.from_numpy(normalized).reshape(N, T, D).to(device)


def invert_scaler_to_original(
    trajectories: torch.Tensor, scaler: StandardScaler
) -> torch.Tensor:
    """Invert a scaler transform back to original simulator units.

    Args:
        trajectories: (N, T, D) normalized tensor.
        scaler: the same StandardScaler used to normalize.
    Returns:
        (N, T, D) tensor in original units on the same device.
    """
    device = trajectories.device
    N, T, D = trajectories.shape
    flat = trajectories.reshape(-1, D).cpu().numpy()
    original = scaler.inverse_transform(flat).astype(np.float32)
    return torch.from_numpy(original).reshape(N, T, D).to(device)

class TrajectoryDataset(Dataset):
    """
    Unified Dataset for loading trajectories from disk.

    Features:
    - Reads index.csv
    - Filters by split ('train', 'val', 'testA', 'testB', 'testC')
    - Filters by role ('train_inner', 'val_inner', 'support', 'query')
    - Loads .npy files into Float32 tensors
    - VALIDATES shapes against the active config to prevent silent mismatches.
    """

    def __init__(
        self,
        index_path: str,
        split: str,
        role: Optional[str] = None,
        root_dir: Optional[str] = None,
        check_shapes: bool = True,
        scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            index_path: Path to the generated index.csv
            split: One of ['train', 'val', 'testA', 'testB', 'testC']
            role: Optional filter.
                  For train: 'train_inner' or 'val_inner'
                  For test: 'support' or 'query'
            root_dir: Optional root directory to prepend to file_path
                      (if None, file_path is used as-is).
            check_shapes: If True, check (T+1, d) against config.
            scaler: Optional pre-fitted StandardScaler. When set, __getitem__
                    returns z-scored trajectories. Fit one via fit_scaler().
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found at {index_path}. "
                f"Run generate_trajectories.py first."
            )

        df = pd.read_csv(index_path)

        # 1. Filter by split
        df = df[df["split"] == split]

        # 2. Filter by role, if provided
        if role is not None:
            df = df[df["role"] == role]

        # 3. Crash early if empty (Fixes silent failure issue)
        if df.empty:
            raise RuntimeError(
                f"No rows found in index for split='{split}', role='{role}'. "
                f"Check your index.csv or generation parameters."
            )

        df = df.reset_index(drop=True)

        self.metadata = df
        self.root_dir = root_dir
        self.check_shapes = check_shapes
        self.scaler = scaler  # StandardScaler or None

        # Cache expected shapes from Config
        # N_steps is number of intervals, so points = N_steps + 1
        self.expected_T = cfg.time_grid.n_steps + 1
        self.expected_dim = cfg.basis.x_dim

    def __len__(self) -> int:
        return len(self.metadata)

    def fit_scaler(self) -> StandardScaler:
        """Fit a per-dimension StandardScaler on every trajectory in this split.

        Uses partial_fit to avoid loading all trajectories into memory at once.
        After calling this, self.scaler is set and __getitem__ returns normalized
        tensors. Returns the fitted scaler (also stored as self.scaler).
        """
        scaler = StandardScaler()
        for i in range(len(self)):
            row = self.metadata.iloc[i]
            rel_path = row["file_path"]
            fpath = (
                os.path.join(self.root_dir, rel_path)
                if self.root_dir is not None
                else rel_path
            )
            traj_np = np.load(fpath)  # (T, D) — raw, bypasses scaler
            scaler.partial_fit(traj_np)
        self.scaler = scaler
        return scaler

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row = self.metadata.iloc[idx]
        rel_path = row["file_path"]

        # Resolve path safely
        if self.root_dir is not None:
            file_path = os.path.join(self.root_dir, rel_path)
        else:
            file_path = rel_path

        # Load Data
        try:
            traj_np = np.load(file_path)  # Expected (T+1, d)
        except Exception as e:
            raise IOError(f"Failed to load file {file_path}: {e}")

        traj = torch.from_numpy(traj_np).float()

        # Apply per-dimension normalization if a scaler has been fitted
        if self.scaler is not None:
            traj_np_norm = self.scaler.transform(traj_np).astype(np.float32)
            traj = torch.from_numpy(traj_np_norm)

        # Optional shape check (Fixes silent dimension mismatch)
        if self.check_shapes:
            if traj.ndim != 2:
                raise ValueError(
                    f"Trajectory at {file_path} has ndim={traj.ndim}, expected 2."
                )
            T, d = traj.shape
            if T != self.expected_T or d != self.expected_dim:
                raise ValueError(
                    f"Trajectory at {file_path} has shape {traj.shape}, "
                    f"expected ({self.expected_T}, {self.expected_dim}). "
                    f"Did you change the config without regenerating data?"
                )

        return traj, row["theta_id"]