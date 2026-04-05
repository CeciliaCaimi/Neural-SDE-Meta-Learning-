# config/base_config.py

from dataclasses import dataclass, field
from typing import Tuple

# ------------------------------
# 1. Time grid
# ------------------------------
@dataclass
class TimeGridConfig:
    T: float = 1.0          # final time
    n_steps: int = 200      # dt = 0.005 (Finer grid for cubic stability)

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

# -----------------------------------
# 2. Basis Config
# -----------------------------------
@dataclass
class BasisConfig:
    x_dim: int = 10
    drift_basis_names: Tuple[str, ...] = (
        "const", "x", "x2", "x3", "sin", "cos",  # Base 6 terms
        "tanh", "x4"                              # Enhanced nonlinearity for 10D
    )
    diffusion_basis_names: Tuple[str, ...] = ("const", "absx", "sqrt_absx")
    diffusion_eps: float = 1e-3

    @property
    def n_drift_basis(self) -> int:
        return len(self.drift_basis_names)

    @property
    def n_diffusion_basis(self) -> int:
        return len(self.diffusion_basis_names)

# ----------------------------------------------
# 3. Distributions over θ (The Manifold)
# ----------------------------------------------
@dataclass
class ThetaDistributionConfig:
    # ---- DRIFT ----
    drift_mean_train: float = 0.0
    drift_std_train: float = 0.5

    # Scaled priors to dampen high-order terms (CRITICAL SAFETY for 10D)
    # Added tanh and x4 for richer nonlinear dynamics in 10D
    drift_scales: Tuple[float, ...] = field(default_factory=lambda: (
        1.0, 0.5, 0.05, 0.005,  # const, x, x2, x3
        0.3, 0.3,                # sin, cos
        0.2, 0.001               # tanh, x4 (very small for x4 stability)
    ))

    # ---- DIFFUSION ----
    diffusion_mean_train: float = -2.0
    diffusion_std_train: float = 1.0

    diffusion_scales: Tuple[float, ...] = field(default_factory=lambda: (
        0.5, 0.2, 0.1
    ))

    # ---- TEST REGIMES ----
    use_same_dist_for_test_A: bool = True

    # Case B: Extrapolation (Mild Shift)
    drift_mean_shift_test_B: float = 0.5
    diffusion_mean_shift_test_B: float = 0.5

    # Case C: Strong Extrapolation
    drift_mean_shift_test_C: float = 1.0
    diffusion_mean_shift_test_C: float = 1.0
    drift_std_scale_test_C: float = 2.0

    drift_std_test: float = 1.0
    diffusion_std_test: float = 1.0

# -------------------------------
# 4. Stability / Safety
# -------------------------------
@dataclass
class StabilityConfig:
    max_state_abs: float = 10.0
    max_retries_per_theta: int = 20

# -------------------------------
# 5. Initialization (RESTORED)
# -------------------------------
@dataclass
class InitializationConfig:
    x0_mean: float = 0.0
    x0_std: float = 0.25  # Reduced from 0.5 for 10D stability

# --------------------------------
# 6. Dataset Sizes
# --------------------------------
@dataclass
class DatasetSizesConfig:
    n_train_thetas: int = 150  # Increased from 100 for 10D
    n_val_thetas: int = 30     # Increased from 20
    n_test_thetas: int = 30    # Increased from 20

    # Trajectories per theta (increased for better learning in 10D)
    n_train_traj_per_theta_train_inner: int = 80  # Increased from 64
    n_train_traj_per_theta_val_inner: int = 20    # Increased from 16
    n_val_traj_per_theta: int = 40                # Increased from 32
    n_test_support_traj_per_theta: int = 10
    n_test_query_traj_per_theta: int = 32

# -----------------
# 7. Latent & Model
# -----------------
@dataclass
class LatentConfig:
    latent_dim: int = 16           # Increased from 8 to capture 10D variability
    encoder_hidden_dim: int = 256  # Increased from 64 for 10D state
    sde_hidden_dim: int = 128      # Increased from 64
    head_hidden_dim: int = 128     # Increased from 64

# --------------
# 8. Paths
# --------------
@dataclass
class DatasetPathsConfig:
    data_root: str = "data/"
    meta_params_path: str = "data/meta_params.npz"
    train_traj_root: str = "data/train_trajectories/"
    val_traj_root: str = "data/val_trajectories/"
    test_traj_root: str = "data/test_trajectories/"

# -------------------------
# 9. Master Config
# -------------------------
@dataclass
class BaseConfig:
    time_grid: TimeGridConfig = field(default_factory=TimeGridConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    theta_dist: ThetaDistributionConfig = field(default_factory=ThetaDistributionConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    init: InitializationConfig = field(default_factory=InitializationConfig)
    dataset_sizes: DatasetSizesConfig = field(default_factory=DatasetSizesConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    paths: DatasetPathsConfig = field(default_factory=DatasetPathsConfig)
    
    device: str = "cuda"
    global_seed: int = 12345

    test_regimes: Tuple[str, ...] = ("A", "B", "C")

cfg = BaseConfig()