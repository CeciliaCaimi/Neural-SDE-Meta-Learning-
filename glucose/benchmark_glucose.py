#glucose/benchmark_glucose.py
import os
import numpy as np
from config.base_config import cfg
from deepmind.gated_finetuning_regularized_dm import main as run_adaptation

# --- CONFIG ---
DATA_DIR = "data/glucose_simple"
CKPT_PATH = "checkpoints/glucose/model_best.pt"

def benchmark():
    print("📈 BENCHMARKING GLUCOSE ADAPTATION")
    
    if not os.path.exists(CKPT_PATH):
        print("❌ Train the model first!")
        return

    # 1. Detect Dimension
    train_dir = os.path.join(DATA_DIR, "train")
    sample = [f for f in os.listdir(train_dir) if f.endswith(".npy")][0]
    x_dim = np.load(os.path.join(train_dir, sample)).shape[-1]

    # 2. Setup Config
    cfg.paths.data_root = DATA_DIR
    cfg.basis.x_dim = x_dim
    
    # 3. Point to Specific Model
    cfg.ckpt_path_override = CKPT_PATH
    
    # 4. Run Your Adaptation Framework
    # It will automatically look for the "test" split in index.csv
    try:
        run_adaptation()
        print("✅ Results generated in 'results/' folder.")
    except Exception as e:
        print(f"❌ Adaptation Failed: {e}")

if __name__ == "__main__":
    benchmark()