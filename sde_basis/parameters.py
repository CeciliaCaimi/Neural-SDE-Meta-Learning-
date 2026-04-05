# sde_basis/parameters.py

from dataclasses import dataclass
import torch


@dataclass
class Theta:
    """
    Container for the *true* SDE parameters.

    Attributes
    ----------
    theta_b : torch.Tensor
        Coefficients for drift basis. Shape: (d, n_drift_basis)
    theta_sigma : torch.Tensor
        Coefficients for diffusion basis. Shape: (d, n_diffusion_basis)
    id : str
        Optional identifier for this parameter set (e.g. 'train_042').
    """
    theta_b: torch.Tensor
    theta_sigma: torch.Tensor
    id: str = ""

    def to(self, device: str) -> "Theta":
        """Move parameter tensors to a given device."""
        return Theta(
            theta_b=self.theta_b.to(device),
            theta_sigma=self.theta_sigma.to(device),
            id=self.id,
        )
