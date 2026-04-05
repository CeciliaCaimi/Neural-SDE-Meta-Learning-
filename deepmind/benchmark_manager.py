#deepmind/benchmark_manager.py
import os
import numpy as np
import torch
from config.base_config import cfg
from deepmind.gated_finetuning_regularized_dm import main as run_adaptation_experiment

# --- CONFIGURATION ---
TASKS = ["reacher", "finger", "cheetah"]
BASE_DATA_DIR = "data/deepmind"
BASE_CKPT_DIR = "checkpoints/deepmind"

def benchmark_task(task_name):
    print(f"\n========================================")
    print(f"🧪 BENCHMARKING: {task_name.upper()}")
    print(f"========================================")
    
    # 1. verify Checkpoint Exists
    # Must match where train_manager saved it
    ckpt_path = os.path.join(BASE_CKPT_DIR, task_name, "model_best.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint missing: {ckpt_path}")
        print(f"   Run 'python -m deepmind.train_manager' first.")
        return

    # 2. Detect Dimension (from .npy data)
    task_data_dir = os.path.join(BASE_DATA_DIR, task_name)
    train_dir = os.path.join(task_data_dir, "train")
    
    try:
        files = [f for f in os.listdir(train_dir) if f.endswith(".npy")]
        sample_path = os.path.join(train_dir, files[0])
        x_dim = np.load(sample_path).shape[-1]
    except Exception as e:
        print(f"❌ Error detecting dimension for {task_name}: {e}")
        return

    # 3. Configure Global Config
    cfg.paths.data_root = task_data_dir
    cfg.basis.x_dim = x_dim
    
    # 4. OVERRIDE CHECKPOINT PATH
    # This tells the adaptation script EXACTLY which file to load
    cfg.ckpt_path_override = ckpt_path
    
    # 5. Run Evaluation
    try:
        run_adaptation_experiment()
        print(f"✅ {task_name} Results Generated.")
    except Exception as e:
        print(f"❌ Benchmark crashed for {task_name}: {e}")

if __name__ == "__main__":
    for task in TASKS:
        benchmark_task(task)