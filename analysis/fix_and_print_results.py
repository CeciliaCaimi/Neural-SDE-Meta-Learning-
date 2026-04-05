import pandas as pd
import os

def main():
    print("🔧 Fixing Column Alignment...")
    
    file_path = "results/gated_regularized_final.csv"
    
    # 1. Read the file WITHOUT the header (skip row 0)
    # The data was written in this order:
    # gate_value, residual_error, adapt_time, mse_rollout, mse_final, nll, regime, theta_id, steps_available
    correct_columns = [
        "gate_value", "residual_error", "adapt_time", 
        "mse_rollout", "mse_final", "nll", 
        "regime", "theta_id", "steps_available"
    ]
    
    try:
        df = pd.read_csv(file_path, header=0, names=correct_columns)
        
        # 2. Verify it looks correct now
        print("\n✅ Sample Data (Fixed):")
        print(df.iloc[0])
        
        # 3. Save the fixed version
        df.to_csv("results/gated_regularized_final_fixed.csv", index=False)
        print("\n💾 Saved fixed data to: results/gated_regularized_final_fixed.csv")
        
        # 4. Print the Final Table
        print("\n" + "="*60)
        print("🏆 FINAL RESULTS (Gated Regularized)")
        print("="*60)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        summary = df.groupby(['regime', 'steps_available'])[[
            'mse_rollout', 'mse_final', 'gate_value', 'residual_error'
        ]].mean()
        print(summary)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()