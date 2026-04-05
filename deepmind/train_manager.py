#deepmind/train_manager.py
import os
import shutil
import numpy as np
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
    train_dir = os.path.join(task_data_dir, "train")
    
    # Ensure destination exists
    os.makedirs(task_ckpt_dir, exist_ok=True)
    
    # 2. Robust Dimension Detection
    # We look for ANY .npy file to determine input size
    if not os.path.exists(train_dir):
        print(f"❌ Missing training data for {task_name}")
        return

    files = [f for f in os.listdir(train_dir) if f.endswith(".npy")]
    if not files:
        print(f"❌ No .npy files found in {train_dir}")
        return

    # Load one file to check shape
    # Shape is typically (Time, Dim) -> e.g. (201, 6)
    sample_path = os.path.join(train_dir, files[0])
    try:
        arr = np.load(sample_path)
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