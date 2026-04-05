# simulators/gym_pendulum.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import os
import copy  # <--- CRITICAL IMPORT

from config.base_config import cfg
from models.encoder import TrajEncoder
from models.neural_sde import NeuralSDE
from models.head import ForecastHead
from training.train_meta import simulate_neural_sde_batch

N_TASKS_TEST = 20
HORIZON = 50
N_SHOTS = 2
ADAPT_STEPS = 50
ADAPT_LR = 1e-2

def pad_state(x):
    # (B,T,3) -> (B,T,10)
    batch, time, _ = x.shape
    pad = torch.zeros(batch, time, 7, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)

def make_pendulum_dataset(n_tasks=20, trajs_per_task=10):
    data, params = [], []
    print(f"Generating {n_tasks} Pendulum Tasks...")
    for _ in range(n_tasks):
        mass = np.random.uniform(0.5, 2.0)
        length = np.random.uniform(0.5, 1.5)
        
        env = gym.make('Pendulum-v1')
        env.unwrapped.m = mass
        env.unwrapped.l = length
        
        task_trajs = []
        for _ in range(trajs_per_task):
            obs, _ = env.reset()
            traj = [obs]
            for _ in range(HORIZON):
                obs, _, done, _, _ = env.step([0.0])
                traj.append(obs)
                if done: break
            # Pad or slice to consistent length
            full_traj = np.array(traj[:HORIZON+1])
            task_trajs.append(full_traj)
            
        data.append(np.stack(task_trajs))
        params.append({'mass': mass, 'length': length})
    return data, params

def main():
    device = torch.device(cfg.device)
    os.makedirs('results', exist_ok=True)
    
    # === LOAD SYNTHETIC MODEL ===
    print("Loading Synthetic Meta-Model...")
    x_dim, z_dim = cfg.basis.x_dim, cfg.latent.latent_dim
    encoder = TrajEncoder(x_dim, z_dim, cfg.latent.encoder_hidden_dim).to(device)
    sde = NeuralSDE(x_dim, z_dim, cfg.latent.sde_hidden_dim).to(device)
    head = ForecastHead(x_dim, z_dim, cfg.latent.head_hidden_dim).to(device)
    
    ckpt = torch.load('checkpoints/meta_epoch_50.pt', map_location=device)
    encoder.load_state_dict(ckpt['encoder']) # Check key names (sometimes 'encoder_state_dict')
    sde.load_state_dict(ckpt['sde']) 
    head.load_state_dict(ckpt['head'])
    
    # FREEZE PHYSICS ENGINE
    encoder.eval()
    sde.eval()
    
    # === GYM DATA ===
    test_tasks, test_params = make_pendulum_dataset(N_TASKS_TEST)
    
    results = []
    gen = torch.Generator(device=device).manual_seed(42)
    
    print("🚀 Running Sim-to-Real Adaptation...")
    for task_id, task_np in enumerate(tqdm(test_tasks)):
        task_t = torch.tensor(task_np, dtype=torch.float32).to(device)
        task_pad = pad_state(task_t)
        support, query = task_pad[:N_SHOTS], task_pad[N_SHOTS:]
        
        # === META APPROACH ===
        # 1. Infer Z (Manifold Localization)
        with torch.no_grad():
            z_star = encoder(support[:, :20]).mean(dim=0)
        
        # 2. Deepcopy Head (CRITICAL FIX)
        # Adapt a CLONE, not the original
        head_ft = copy.deepcopy(head)
        head_ft.train()
        
        head_opt = optim.Adam(head_ft.parameters(), lr=ADAPT_LR)
        
        # 3. Adaptation Loop
        for _ in range(ADAPT_STEPS):
            head_opt.zero_grad()
            z_rep = z_star.unsqueeze(0).expand(support.shape[0], -1)
            
            # Simulate (Physics Engine is frozen/shared)
            traj_pred = simulate_neural_sde_batch(sde, support[:,0], z_rep, 
                                                cfg.time_grid.T, cfg.time_grid.n_steps, 
                                                cfg.stability.max_state_abs, gen)
            
            # Predict with ADAPTED head
            pred_final = head_ft(traj_pred[:,-1], z_rep)
            loss = F.mse_loss(pred_final, support[:,-1])
            loss.backward()
            head_opt.step()
        
        # 4. Meta Eval
        head_ft.eval()
        with torch.no_grad():
            z_q = z_star.unsqueeze(0).expand(query.shape[0], -1)
            traj_q = simulate_neural_sde_batch(sde, query[:,0], z_q, 
                                             cfg.time_grid.T, cfg.time_grid.n_steps, 
                                             cfg.stability.max_state_abs, gen)
            pred_q_meta = head_ft(traj_q[:,-1], z_q)
            mse_meta = F.mse_loss(pred_q_meta, query[:,-1]).item()
        
        # === SCRATCH APPROACH ===
        # Naive Regressor (Weak Baseline)
        head_scratch = ForecastHead(x_dim, z_dim, 32).to(device)
        opt_scr = optim.Adam(head_scratch.parameters(), lr=3e-4, weight_decay=1e-2)
        z_zero = torch.zeros(support.shape[0], z_dim, device=device)
        
        head_scratch.train()
        for _ in range(ADAPT_STEPS):
            opt_scr.zero_grad()
            pred_scr = head_scratch(support[:,0], z_zero)
            loss_scr = F.mse_loss(pred_scr, support[:,-1])
            loss_scr.backward()
            opt_scr.step()
        
        head_scratch.eval()
        with torch.no_grad():
            z_q_zero = torch.zeros(query.shape[0], z_dim, device=device)
            pred_q_scratch = head_scratch(query[:,0], z_q_zero)
            mse_scratch = F.mse_loss(pred_q_scratch, query[:,-1]).item()
        
        results.append({
            'mass': test_params[task_id]['mass'],
            'length': test_params[task_id]['length'],
            'MSE_Meta': mse_meta,
            'MSE_Scratch': mse_scratch
        })
    
    df = pd.DataFrame(results)
    df.to_csv('results/gym_pendulum.csv', index=False)
    print("\n🏆 FINAL RESULTS (Gym Pendulum):")
    print(df[['MSE_Meta', 'MSE_Scratch']].mean())

if __name__ == "__main__":
    main()
