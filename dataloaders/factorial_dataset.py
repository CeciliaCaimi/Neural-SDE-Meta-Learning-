# dataloaders/factorial_dataset.py
#
# Item 6, phase 1a: make the item-5 factorial drift/diffusion benchmark
# (data/factorial_meta_params.npz, data/factorial_trajectories/,
# data/factorial_index.csv) loadable by theta_id, the way
# dataloaders/trajectory_datasets.py's TrajectoryDataset loads the legacy
# train/val/test<Regime> data.
#
# Why this is a NEW module instead of extending TrajectoryDataset: the two
# index schemas are different in a way that matters. The legacy index.csv
# has a `split` column and a `file_path` column per row (one row per
# trajectory file). data/factorial_index.csv (data_gen/generate_factorial_benchmark.py)
# has no `split` column (factorial tasks are not pre-assigned to
# train/val/test -- that assignment is what data_gen/compositional_splits.py
# does, per holdout fraction and seed) and no `file_path` column (one row
# per (theta_id, role) PAIR with an n_achieved count, not one row per file;
# file paths follow a fixed convention
# `<traj_root>/<theta_id>/<role>/traj_<k>.npy` for k in
# range(n_achieved) -- see generate_factorial_benchmark.py's simulate_task).
# Reusing TrajectoryDataset as-is would require it to understand both
# schemas; a second, parallel Dataset class matching its *interface*
# (`.metadata` a DataFrame with a `theta_id` column, `__getitem__` ->
# (tensor, theta_id), `fit_scaler()`) is simpler and leaves
# TrajectoryDataset (and everything that depends on its exact behavior)
# completely untouched.

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data_gen.mechanism_library import MechanismTheta


def load_mechanism_theta_list(meta_params_path: str) -> List[MechanismTheta]:
    """Load the full factorial task list (in on-disk order) from a
    `{"factorial": List[MechanismTheta]}` .npz/.pt file, as saved by
    data_gen/generate_factorial_benchmark.py's save_tasks()."""
    data = torch.load(meta_params_path, map_location="cpu", weights_only=False)
    return list(data["factorial"])


def load_mechanism_theta_dict(meta_params_path: str) -> Dict[str, MechanismTheta]:
    """Same as load_mechanism_theta_list, keyed by theta.id."""
    return {t.id: t for t in load_mechanism_theta_list(meta_params_path)}


class FactorialTrajectoryDataset(Dataset):
    """
    Loads trajectories for an explicit list of factorial task ids and one
    role, from a factorial-style index csv (theta_id, role, n_achieved, ...)
    plus the fixed `<traj_root>/<theta_id>/<role>/traj_<k>.npy` file-naming
    convention.

    Matches dataloaders.trajectory_datasets.TrajectoryDataset's *interface*
    (`.metadata` DataFrame with a `theta_id` column, `__getitem__` ->
    (Tensor, theta_id), `fit_scaler()`) so existing per-task helpers that
    only use that interface (e.g.
    adaptation/gated_finetuning_regularized.get_task_data) work unchanged
    against it -- see FACTORIAL_BENCHMARK.md / COMPOSITIONAL_EXPERIMENT.md
    for why a parallel class, not an extension of TrajectoryDataset.
    """

    def __init__(
        self,
        index_path: str,
        traj_root: str,
        role: str,
        task_ids: Optional[Sequence[str]] = None,
        check_shapes: bool = True,
        expected_dim: Optional[int] = None,
        expected_T: Optional[int] = None,
        scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            index_path: path to a factorial-style index csv (e.g.
                data/factorial_index.csv or the index written by
                data_gen/generate_factorial_training_trajectories.py).
            traj_root: root directory the index's theta_id/role subdirs live
                under (e.g. data/factorial_trajectories or
                data/factorial_train_trajectories).
            role: a single role to filter to (e.g. "support", "query",
                "train_inner", "val_inner").
            task_ids: explicit list of theta_ids to include. If None, every
                task_id present for `role` in the index is included -- but
                an explicit list is the expected normal usage (item 6 phase
                1d: training/adaptation entry points take explicit task id
                lists, not regime names).
            expected_dim / expected_T: if given (and check_shapes=True),
                validate every loaded trajectory's shape against them. If
                omitted, falls back to cfg.basis.x_dim / cfg.time_grid.n_steps+1
                (matching TrajectoryDataset's default behavior), read lazily
                so toy-scale cfg overrides applied before this call are
                respected.
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Factorial index not found at {index_path}.")

        df = pd.read_csv(index_path)
        df = df[df["role"] == role]
        if task_ids is not None:
            task_ids = list(task_ids)
            df = df[df["theta_id"].isin(task_ids)]

        if df.empty:
            raise RuntimeError(
                f"No rows found in {index_path} for role={role!r}"
                + (f", task_ids={task_ids!r}" if task_ids is not None else "") + "."
            )

        rows = []
        for _, row in df.iterrows():
            theta_id = row["theta_id"]
            n_achieved = int(row["n_achieved"])
            for k in range(n_achieved):
                rows.append({
                    "theta_id": theta_id,
                    "file_path": os.path.join(traj_root, theta_id, role, f"traj_{k}.npy"),
                })

        if not rows:
            raise RuntimeError(
                f"role={role!r} matched tasks in {index_path} but every matched "
                f"task has n_achieved=0 -- no trajectory files to load."
            )

        self.metadata = pd.DataFrame(rows)
        self.role = role
        self.check_shapes = check_shapes
        self.scaler = scaler
        self._expected_dim = expected_dim
        self._expected_T = expected_T

    def _shape_expectations(self):
        if self._expected_dim is not None and self._expected_T is not None:
            return self._expected_T, self._expected_dim
        from config.base_config import cfg
        return (
            self._expected_T if self._expected_T is not None else cfg.time_grid.n_steps + 1,
            self._expected_dim if self._expected_dim is not None else cfg.basis.x_dim,
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def fit_scaler(self) -> StandardScaler:
        """Fit a per-dimension StandardScaler over every trajectory in this
        dataset. Mirrors TrajectoryDataset.fit_scaler()."""
        scaler = StandardScaler()
        for i in range(len(self)):
            fpath = self.metadata.iloc[i]["file_path"]
            traj_np = np.load(fpath)
            scaler.partial_fit(traj_np)
        self.scaler = scaler
        return scaler

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        file_path = row["file_path"]
        try:
            traj_np = np.load(file_path)
        except Exception as e:
            raise IOError(f"Failed to load file {file_path}: {e}")

        traj = torch.from_numpy(traj_np).float()

        if self.scaler is not None:
            traj_np_norm = self.scaler.transform(traj_np).astype(np.float32)
            traj = torch.from_numpy(traj_np_norm)

        if self.check_shapes:
            if traj.ndim != 2:
                raise ValueError(f"Trajectory at {file_path} has ndim={traj.ndim}, expected 2.")
            expected_T, expected_dim = self._shape_expectations()
            T, d = traj.shape
            if T != expected_T or d != expected_dim:
                raise ValueError(
                    f"Trajectory at {file_path} has shape {traj.shape}, "
                    f"expected ({expected_T}, {expected_dim})."
                )

        return traj, row["theta_id"]


def task_ids_for_role(index_path: str, role: str) -> List[str]:
    """Convenience: all theta_ids with at least one usable (n_achieved > 0)
    trajectory for `role` in a factorial-style index csv."""
    df = pd.read_csv(index_path)
    df = df[(df["role"] == role) & (df["n_achieved"] > 0)]
    return sorted(df["theta_id"].unique().tolist())
