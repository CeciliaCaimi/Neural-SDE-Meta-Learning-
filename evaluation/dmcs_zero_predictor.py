"""DMCS Zero-Predictor Sanity Check — Final Validation Exp 1.

For each DMCS task, evaluate trivial baseline x_hat = 0 on the same
standardised trajectories. Report MSE ratio = MSE(Model C) / MSE(Zero).

Usage: cd ~/fresh-run && PYTHONPATH=. python evaluation/dmcs_zero_predictor.py
"""
import os, sys, torch, numpy as np, pandas as pd
sys.path.append(os.getcwd())

from config.base_config import cfg

def main():
    data_dir = "data/test_trajectories"
    results = []

    for regime in ['testA', 'testB', 'testC']:
        theta_ids = sorted([d for d in os.listdir(data_dir) if d.startswith(regime)])

        for theta_id in theta_ids:
            task_dir = f"{data_dir}/{theta_id}"
            qry_files = sorted(os.listdir(f"{task_dir}/query"))

            for f in qry_files[:2]:
                query = np.load(f"{task_dir}/query/{f}")  # (T, D)
                # Zero predictor: predict 0 at every step
                mse_zero = np.mean(query ** 2)
                results.append({'regime': regime, 'theta_id': theta_id, 'mse_zero': mse_zero})

    df = pd.DataFrame(results)

    # Load Model C results
    model_c = pd.read_csv('results/gated_regularized_final_fixed.csv')
    model_c_201 = model_c[model_c['steps_available'] == 201]

    print("=== DMCS Zero-Predictor Sanity Check ===\n")
    print(f"{'Regime':<8} {'MSE(Zero)':<12} {'MSE(Model C)':<14} {'Ratio (C/Zero)':<14}")
    print("-" * 50)

    for regime in ['testA', 'testB', 'testC']:
        mse_zero_mean = df[df['regime'] == regime]['mse_zero'].mean()
        mc_rows = model_c_201[model_c_201['theta_id'].str.startswith(regime)]
        mse_mc_mean = mc_rows['mse_rollout'].mean() if len(mc_rows) > 0 else float('nan')
        ratio = mse_mc_mean / mse_zero_mean if mse_zero_mean > 0 else float('nan')
        print(f"{regime:<8} {mse_zero_mean:<12.4f} {mse_mc_mean:<14.4f} {ratio:<14.4f}")

    os.makedirs('results', exist_ok=True)
    df.to_csv('results/dmcs_zero_predictor.csv', index=False)
    print(f"\n✅ Saved results/dmcs_zero_predictor.csv")

if __name__ == '__main__':
    main()
