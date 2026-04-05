#baselines/models_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticGRU(nn.Module):
    """
    A GRU that outputs a Gaussian distribution (mean, variance) at every step.
    Enables NLL (Uncertainty) and Calibration metrics.
    """
    def __init__(self, x_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.x_dim = x_dim
        
        # 1. Standard GRU Backbone
        self.gru = nn.GRU(
            input_size=x_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 2. Probabilistic Heads
        self.mu_head = nn.Linear(hidden_dim, x_dim)
        self.logvar_head = nn.Linear(hidden_dim, x_dim)  # Predict log(variance) for stability

    def forward(self, x, h=None):
        # x: (Batch, Time, Dim)
        # output: (Batch, Time, Hidden), h_n
        features, h_n = self.gru(x, h)
        
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        # Enforce variance positivity and numerical stability
        # var = exp(logvar) + epsilon
        var = torch.exp(logvar) + 1e-6
        
        return mu, var, h_n