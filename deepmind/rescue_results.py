import pandas as pd
import os
import glob

# SETTINGS
RESULTS_DIR = "results"

def universal_rescue():
    print("🚑 RUNNING UNIVERSAL DATA DECODER...\n")
    
    # Get all result files
    files = glob.glob(os.path.join(RESULTS_DIR, "dm_*_model_c.csv"))
    
    if not files:
        print("❌ No CSV files found.")
        return

    for f in files:
        task = os.path.basename(f).replace("dm_", "").replace("_model_c.csv", "").upper()
        print(f"\n{'='*50}")
        print(f"📂 ANALYZING FILE: {task}")
        print(f"{'='*50}")
        
        try:
            df = pd.read_csv(f)
            
            # Separate columns by type
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            print(f"   found text cols: {text_cols}")
            print(f"   found num cols:  {num_cols}")
            
            # STRATEGY: Try grouping by EVERY text column and averaging EVERY number column
            # One of these combinations is the answer.
            
            for t_col in text_cols:
                # We only care about columns that look like 'split' (contain 'testA', 'testB')
                unique_vals = df[t_col].unique()
                if len(unique_vals) > 10: continue # Skip columns with too many unique values (like IDs)
                
                print(f"\n🔹 GROUPING BY: '{t_col}' (Values: {unique_vals[:3]}...)")
                
                # Calculate mean for all numeric columns
                summary = df.groupby(t_col)[num_cols].mean()
                print(summary)
                print("-" * 30)

        except Exception as e:
            print(f"❌ Error reading {task}: {e}")

if __name__ == "__main__":
    universal_rescue()