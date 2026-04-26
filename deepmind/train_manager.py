#deepmind/train_manager.py
import os
import shutil
import pandas as pd
import torch
from config.base_config import cfg
from training.train_meta import train_meta_loop

# --- CONFIGURATION ---
TASKS = ["reacher", "finger", "cheetah"]
BASE_DATA_DIR = "data/deepmind"
BASE_CKPT_DIR = "checkpoints/deepmind"

# The file your training script generates by default
GENERIC_OUTPUT_CKPT = "checkpoints/meta_epoch_50.pt"

def train_task_model(task_name):
    print(f"\n========================================")
    print(f"💪 STARTING TRAINING: {task_name.upper()}")
    print(f"========================================")
    
    # 1. Prepare Paths
    task_data_dir = os.path.join(BASE_DATA_DIR, task_name)
    task_ckpt_dir = os.path.join(BASE_CKPT_DIR, task_name)

    # Ensure destination exists
    os.makedirs(task_ckpt_dir, exist_ok=True)
    
    # 2. Robust Dimension Detection
    # Use the index.csv written by generate_all.py to find a training sample.
    index_path = os.path.join(task_data_dir, "index.csv")
    if not os.path.exists(index_path):
        print(f"❌ Missing index.csv for {task_name} at {index_path}")
        return

    index = pd.read_csv(index_path)
    train_rows = index[index["split"] == "train"]
    if train_rows.empty:
        print(f"❌ No training rows found in index for {task_name}")
        return

    # Load one .pt tensor to detect the state dimension.
    # Shape is (N_shots, Time, Dim) for support or (1, Time, Dim) for query.
    sample_path = train_rows["path"].iloc[0]
    try:
        arr = torch.load(sample_path, weights_only=False)
        x_dim = arr.shape[-1]
    except Exception as e:
        print(f"❌ Error reading {sample_path}: {e}")
        return

    print(f"   📍 Data Source: {task_data_dir}")
    print(f"   📍 Output Dest: {task_ckpt_dir}/model_best.pt")
    print(f"   📍 Detected Dim: {x_dim}")

    # 3. Configure Global Config
    cfg.paths.data_root = task_data_dir
    cfg.basis.x_dim = x_dim
    
    # Reset/Clear any previous generic checkpoint to ensure we don't move an old file
    if os.path.exists(GENERIC_OUTPUT_CKPT):
        os.remove(GENERIC_OUTPUT_CKPT)

    # 4. Run Training
    try:
        train_meta_loop()
    except Exception as e:
        print(f"❌ Training crashed for {task_name}: {e}")
        return

    # 5. SAFEGUARD: Move and Rename Checkpoint
    # This ensures Reacher doesn't get overwritten by Finger
    final_path = os.path.join(task_ckpt_dir, "model_best.pt")
    
    if os.path.exists(GENERIC_OUTPUT_CKPT):
        shutil.move(GENERIC_OUTPUT_CKPT, final_path)
        print(f"✅ SUCCESS: Model saved to {final_path}")
    else:
        print(f"❌ ERROR: Expected checkpoint {GENERIC_OUTPUT_CKPT} was not created!")

if __name__ == "__main__":
    for task in TASKS:
        train_task_model(task)