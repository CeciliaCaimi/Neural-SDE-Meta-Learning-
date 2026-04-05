# dataloaders/trajectory_datasets.py

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from typing import Optional, Tuple

# We import cfg to sanity-check shapes against the global configuration
from config.base_config import cfg

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

        # Cache expected shapes from Config
        # N_steps is number of intervals, so points = N_steps + 1
        self.expected_T = cfg.time_grid.n_steps + 1
        self.expected_dim = cfg.basis.x_dim

    def __len__(self) -> int:
        return len(self.metadata)

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