"""Harder Regime Switch Plot — Priority 3.1.

Plots MSE rollout vs scale factor to show graceful degradation
when tasks are pushed further from the training distribution.

Inputs:  results/harder_regime_switch.csv
Outputs: results/harder_regime_switch_plot.png
"""
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INPUT_CSV = "results/harder_regime_switch.csv"
OUTPUT_PLOT = "results/harder_regime_switch_plot.png"

def main():
    df = pd.read_csv(INPUT_CSV)
    steps_list = sorted(df['steps'].unique())

    fig, axes = plt.subplots(1, len(steps_list), figsize=(6 * len(steps_list), 5))
    if len(steps_list) == 1:
        axes = [axes]

    for ax, steps in zip(axes, steps_list):
        sub = df[df['steps'] == steps]
        means = sub.groupby('scale')['mse_rollout'].agg(['mean', 'std']).reset_index()
        ax.errorbar(means['scale'], means['mean'], yerr=means['std'],
                    marker='o', capsize=5, linewidth=2, markersize=8, color='#2196F3')
        ax.set_xlabel('Scale Factor (distance from training)', fontsize=12)
        ax.set_ylabel('MSE Rollout', fontsize=12)
        ax.set_title(f'Harder Regime Switch (steps={steps})', fontsize=13)
        ax.set_xticks(means['scale'].tolist())
        ax.grid(True, alpha=0.3)

        # Annotate degradation factor
        baseline = means[means['scale'] == 1.0]['mean'].values[0]
        for _, row in means.iterrows():
            if row['scale'] > 1.0:
                factor = row['mean'] / baseline
                ax.annotate(f'{factor:.1f}×', (row['scale'], row['mean']),
                           textcoords="offset points", xytext=(0, 12),
                           ha='center', fontsize=9, color='#666')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"✅ Saved {OUTPUT_PLOT}")

if __name__ == '__main__':
    main()
