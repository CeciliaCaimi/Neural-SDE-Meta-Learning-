#Changes: Updated config imports to match your BaseConfig structure. Implements Case A/B/C logic.

# data_gen/generate_meta_params.py

import os
import torch
from typing import Dict, List, Tuple, Union

# --- CONFIG & MODULES ---
from config.base_config import cfg
from sde_basis.parameters import Theta

def sample_theta(
    d: int,
    n_drift: int,
    n_diff: int,
    drift_mean: Union[float, torch.Tensor], # Can now be a Tensor
    drift_std: float,
    diff_mean: float,
    diff_std: float,
    drift_scales: Tuple[float, ...],
    diff_scales: Tuple[float, ...],
    generator: torch.Generator,
    theta_id: str
) -> Theta:
    """
    Sample one parameter vector θ.
    """
    # Safety Check
    assert len(drift_scales) == n_drift
    assert len(diff_scales) == n_diff

    # 1. Sample Raw Gaussians
    raw_b = torch.randn(d, n_drift, generator=generator)
    raw_sigma = torch.randn(d, n_diff, generator=generator)

    # 2. Basis Scaling
    scale_b = torch.tensor(drift_scales, dtype=raw_b.dtype).unsqueeze(0)
    scale_sigma = torch.tensor(diff_scales, dtype=raw_sigma.dtype).unsqueeze(0)

    # 3. Apply Mean/Std
    # Note: drift_mean can be a vector now. Broadcasting handles it.
    theta_b = (raw_b * drift_std + drift_mean) * scale_b
    theta_sigma = (raw_sigma * diff_std + diff_mean) * scale_sigma

    return Theta(theta_b=theta_b, theta_sigma=theta_sigma, id=theta_id)

def generate_all_meta_params() -> None:
    print("Generating Stabilized Meta-Parameters...")
    
    gen = torch.Generator()
    gen.manual_seed(cfg.global_seed)

    meta_params: Dict[str, List[Theta]] = {}

    d = cfg.basis.x_dim                              
    n_b = cfg.basis.n_drift_basis             
    n_s = cfg.basis.n_diffusion_basis         

    # --- STABILITY MASK ---
    # We want to shift the distribution for Test B/C, but NOT for x (idx 1) and x^3 (idx 3).
    # Shifting those positively causes exponential explosions.
    # Mask: 1 = Shift, 0 = No Shift.
    # Indices: 0:const, 1:x, 2:x2, 3:x3, 4:sin, 5:cos
    drift_shift_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])


    # --- 1. Train ---
    train_thetas: List[Theta] = []
    for i in range(cfg.dataset_sizes.n_train_thetas): 
        t = sample_theta(
            d, n_b, n_s,
            drift_mean=cfg.theta_dist.drift_mean_train,
            drift_std=cfg.theta_dist.drift_std_train,
            diff_mean=cfg.theta_dist.diffusion_mean_train,
            diff_std=cfg.theta_dist.diffusion_std_train,
            drift_scales=cfg.theta_dist.drift_scales,
            diff_scales=cfg.theta_dist.diffusion_scales,
            generator=gen,
            theta_id=f"train_{i:03d}"
        )
        train_thetas.append(t)
    meta_params["train"] = train_thetas

    # --- 2. Validation ---
    val_thetas: List[Theta] = []
    for i in range(cfg.dataset_sizes.n_val_thetas):
        t = sample_theta(
            d, n_b, n_s,
            drift_mean=cfg.theta_dist.drift_mean_train,
            drift_std=cfg.theta_dist.drift_std_train,
            diff_mean=cfg.theta_dist.diffusion_mean_train,
            diff_std=cfg.theta_dist.diffusion_std_train,
            drift_scales=cfg.theta_dist.drift_scales,
            diff_scales=cfg.theta_dist.diffusion_scales,
            generator=gen,
            theta_id=f"val_{i:03d}"
        )
        val_thetas.append(t)
    meta_params["val"] = val_thetas

    # --- 3. Test Regimes ---
    for regime in cfg.test_regimes:
        test_list: List[Theta] = []
        n_test = cfg.dataset_sizes.n_test_thetas

        # Defaults
        b_mean_base = cfg.theta_dist.drift_mean_train
        s_mean_base = cfg.theta_dist.diffusion_mean_train
        b_std = cfg.theta_dist.drift_std_train
        s_std = cfg.theta_dist.diffusion_std_test

        # Calculate Shift Vectors
        if regime == "A":
            # No Shift
            drift_mean_vec = b_mean_base
            diff_mean_vec = s_mean_base
        
        elif regime == "B":
            # Mild Shift (Masked)
            shift_b = cfg.theta_dist.drift_mean_shift_test_B
            shift_s = cfg.theta_dist.diffusion_mean_shift_test_B
            
            # Apply shift only to stable terms
            drift_mean_vec = b_mean_base + (shift_b * drift_shift_mask)
            diff_mean_vec = s_mean_base + shift_s # Shift all diffusion terms
            
            b_std = cfg.theta_dist.drift_std_test

        elif regime == "C":
            # Strong Shift (Masked)
            shift_b = cfg.theta_dist.drift_mean_shift_test_C
            shift_s = cfg.theta_dist.diffusion_mean_shift_test_C
            
            # Apply shift only to stable terms
            drift_mean_vec = b_mean_base + (shift_b * drift_shift_mask)
            diff_mean_vec = s_mean_base + shift_s
            
            b_std = cfg.theta_dist.drift_std_train * cfg.theta_dist.drift_std_scale_test_C

        else:
            raise ValueError(f"Unknown regime {regime}")

        for i in range(n_test):
            t = sample_theta(
                d, n_b, n_s,
                drift_mean=drift_mean_vec, # Passing vector now
                drift_std=b_std,
                diff_mean=diff_mean_vec,
                diff_std=s_std,
                drift_scales=cfg.theta_dist.drift_scales,
                diff_scales=cfg.theta_dist.diffusion_scales,
                generator=gen,
                theta_id=f"test{regime}_{i:03d}"
            )
            test_list.append(t)
        meta_params[f"test{regime}"] = test_list

    # 4. Save
    os.makedirs(os.path.dirname(cfg.paths.meta_params_path), exist_ok=True)
    torch.save(meta_params, cfg.paths.meta_params_path)
    print(f"Saved stabilized meta-params to {cfg.paths.meta_params_path}")
    
    # Summary
    print(f"Generated: {len(train_thetas)} Train, {len(val_thetas)} Val")
    for r in cfg.test_regimes:
        print(f"Generated {len(meta_params[f'test{r}'])} for Test{r}")

if __name__ == "__main__":
    generate_all_meta_params()