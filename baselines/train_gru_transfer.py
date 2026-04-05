#baselines/train_gru.py
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.base_config import cfg
from dataloaders.trajectory_datasets import TrajectoryDataset
from baselines.models_gru import ProbabilisticGRU

# CONFIG
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3

def train_gru_transfer():
    device = torch.device(cfg.device)
    print("📉 Baseline: Pre-Training Probabilistic GRU (Warm-Start)...")
    
    # 1. Data (All Training Tasks)
    index_path = os.path.join(cfg.paths.data_root, "index.csv")
    train_ds = TrajectoryDataset(index_path, "train", "train_inner")
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model
    model = ProbabilisticGRU(x_dim=cfg.basis.x_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 3. Train Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_nll = 0
        total_mse = 0
        
        for batch, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            
            # Input: x_{0:T-1}, Target: x_{1:T}
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :]
            
            # Forward
            mu, var, _ = model(inputs)
            
            # Loss: Gaussian NLL
            # loss = 0.5 * (log(var) + (target - mu)^2 / var)
            loss_nll = F.gaussian_nll_loss(mu, targets, var, reduction='mean')
            
            optimizer.zero_grad()
            loss_nll.backward()
            optimizer.step()
            
            total_nll += loss_nll.item()
            total_mse += F.mse_loss(mu, targets).item()
            
        print(f"Epoch {epoch} | NLL: {total_nll/len(loader):.4f} | MSE: {total_mse/len(loader):.4f}")
        
    # 4. Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/gru_warmstart.pt")
    print("✅ GRU Warm-Start Checkpoint Saved.")

if __name__ == "__main__":
    train_gru_transfer()