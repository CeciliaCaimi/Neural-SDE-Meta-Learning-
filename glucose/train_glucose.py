#glucose/train_glucose.py
import os
import shutil
import numpy as np
from config.base_config import cfg
from training.train_meta import train_meta_loop

# --- CONFIG ---
DATA_DIR = "data/glucose_simple"
CKPT_DIR = "checkpoints/glucose"
GENERIC_CKPT = "checkpoints/meta_epoch_50.pt"

def train():
    print("💉 STARTING GLUCOSE TRAINING")
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    # 1. Detect Dimension from first file
    train_dir = os.path.join(DATA_DIR, "train")
    sample = [f for f in os.listdir(train_dir) if f.endswith(".npy")][0]
    x_dim = np.load(os.path.join(train_dir, sample)).shape[-1]
    print(f"   Detected Dimension: {x_dim}")

    # 2. Setup Config
    cfg.paths.data_root = DATA_DIR
    cfg.basis.x_dim = x_dim
    
    # Remove old junk
    if os.path.exists(GENERIC_CKPT): os.remove(GENERIC_CKPT)

    # 3. Train
    try:
        train_meta_loop()
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        return

    # 4. Save Safely
    final_path = os.path.join(CKPT_DIR, "model_best.pt")
    if os.path.exists(GENERIC_CKPT):
        shutil.move(GENERIC_CKPT, final_path)
        print(f"✅ Model saved to: {final_path}")
    else:
        print("❌ Error: Checkpoint not created.")

if __name__ == "__main__":
    train()