# baselines/compare_baselines.py
# To run: python -m baselines.compare_baselines

import os
import pandas as pd

def main():
    meta_path = "results/adaptation_results.csv"
    transfer_path = "results/transfer_baseline_results.csv"
    
    if not os.path.exists(meta_path):
        print(f"❌ Missing: {meta_path}")
        return
    
    if not os.path.exists(transfer_path):
        print(f"❌ Missing: {transfer_path}")
        return
    
    df_meta = pd.read_csv(meta_path)
    df_transfer = pd.read_csv(transfer_path)
    
    print("\n" + "="*80)
    print("📊 BASELINE COMPARISON: Meta-Learning vs Transfer Learning")
    print("="*80)
    
    # PATH MSE
    print("\n📈 PATH MSE (Full Trajectory Physics)")
    print("-"*80)
    print("\nMeta-Learning (Zero-Shot vs Few-Shot):")
    print(df_meta.groupby("regime")[["mse_path_zeroshot", "mse_path_fewshot"]].mean())
    
    print("\nTransfer Learning (Zero-Shot vs Fine-Tuned):")
    print(df_transfer.groupby("regime")[["mse_path_zeroshot", "mse_path_transfer"]].mean())
    
    # HEAD MSE
    print("\n" + "="*80)
    print("📈 HEAD MSE (Final-Step Forecasting)")
    print("-"*80)
    print("\nMeta-Learning (Zero-Shot vs Few-Shot):")
    print(df_meta.groupby("regime")[["mse_head_zeroshot", "mse_head_fewshot"]].mean())
    
    print("\nTransfer Learning (Zero-Shot vs Fine-Tuned):")
    print(df_transfer.groupby("regime")[["mse_head_zeroshot", "mse_head_transfer"]].mean())
    
    # Simple winner summary
    print("\n" + "="*80)
    print("🏆 WINNER ANALYSIS (by FINAL-STEP MSE)")
    print("="*80)
    
    for regime in ["testA", "testB", "testC"]:
        meta_fs = df_meta[df_meta["regime"] == regime]["mse_head_fewshot"].mean()
        transfer_ft = df_transfer[df_transfer["regime"] == regime]["mse_head_transfer"].mean()
        
        if pd.isna(meta_fs) or pd.isna(transfer_ft):
            print(f"\n{regime}: missing data")
            continue
        
        winner = "Meta-Learning" if meta_fs < transfer_ft else "Transfer Learning"
        margin = abs(meta_fs - transfer_ft)
        pct = (margin / max(meta_fs, transfer_ft)) * 100
        
        print(f"\n{regime}:")
        print(f"  Meta-Learning (Few-Shot):        {meta_fs:.6f}")
        print(f"  Transfer Learning (Fine-Tuned):  {transfer_ft:.6f}")
        print(f"  Winner: {winner} (Δ≈{pct:.1f}% )")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
