import os
import torch
import pandas as pd

ROOT_DIR = "checkpoints"

def check_model_dimension(path):
    try:
        # Load checkpoint on CPU to be fast
        ckpt = torch.load(path, map_location="cpu")
        
        # We need to find the Encoder's input weight
        # Common keys: 'encoder_state_dict', 'encoder', 'model_state_dict'
        state_dict = None
        if 'encoder_state_dict' in ckpt:
            state_dict = ckpt['encoder_state_dict']
        elif 'encoder' in ckpt:
            state_dict = ckpt['encoder']
        
        if state_dict is None:
            return "Unknown (No Encoder)"
            
        # Look for the RNN input weight
        # Usually named: 'rnn.weight_ih_l0' or 'net.0.weight'
        weight = None
        for key in state_dict.keys():
            if 'weight_ih' in key: # GRU/RNN input weight
                weight = state_dict[key]
                break
            if '0.weight' in key and 'weight_hh' not in key: # Linear input
                weight = state_dict[key]
                break
        
        if weight is None:
            return "Unknown (Layer not found)"
            
        # The shape is usually [Hidden, Input_Dim]
        # So shape[1] is the input dimension
        return f"{weight.shape[1]}D"
        
    except Exception as e:
        return f"Error: {str(e)[:20]}"

def scan_folder():
    print(f"🕵️‍♂️ Scanning {ROOT_DIR} for surviving models...\n")
    
    results = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".pt"):
                full_path = os.path.join(root, file)
                dim = check_model_dimension(full_path)
                results.append({"Path": full_path, "Dimension": dim})

    if not results:
        print("❌ No checkpoint files found!")
        return

    # Print clean table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\n💡 GUIDE:")
    print("   - 6D  = Reacher")
    print("   - 9D  = Finger")
    print("   - 10D = YOUR ORIGINAL SETUP (The one you are looking for)")
    print("   - 17D = Cheetah")

if __name__ == "__main__":
    scan_folder()