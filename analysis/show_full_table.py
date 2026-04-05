import pandas as pd
import os

# Formatting to ensure the full table prints
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    file_path = "results/gated_metrics_full.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}")
        return

    print(f"📊 Loading results from {file_path}...\n")
    df = pd.read_csv(file_path)
    
    # Use the correct column name 'steps_available'
    # We also check the metric names to match the CSV exactly
    summary = df.groupby(["regime", "steps_available"])[
        ["mse_rollout", "mse_final", "nll", "adapt_time", "gate"]
    ].mean()
    
    print(summary)
    print("\n✅ Done. This table contains all your experimental results.")

if __name__ == "__main__":
    main()