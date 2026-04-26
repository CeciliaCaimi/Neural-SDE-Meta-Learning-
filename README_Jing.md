# Reproduction Guide
**For: Jing**
**Project:** Meta-Learning for Continuous SDE Manifolds (Cecilia Caimi, MSc Dissertation)

Run all commands from the project root (`DISSERTATION/`). Steps must be followed in order within each part.

---

## Environment

```bash
pip install torch numpy pandas tqdm scikit-learn matplotlib seaborn
pip install dm-control   # DeepMind experiments only

python -c "import torch; print(torch.cuda.is_available())"
```

scikit-learn is required for the normalization pipeline.

---

## Part 1: Synthetic SDE Experiments

### Step 1 - Generate Data

```bash
python -m data_gen.generate_meta_params
python -m data_gen.generate_trajectories
```

Outputs: `data/index.csv`, `data/train_trajectories/`, `data/val_trajectories/`, `data/test_trajectories/`

---

### Step 2 - Meta-Train the Main Model

```bash
python -m training.train_meta
```

Saves checkpoints every 10 epochs to `checkpoints/meta_epoch_{10,20,...,50}.pt`. The primary checkpoint used downstream is `meta_epoch_50.pt`.

The training loop fits a per-dimension `StandardScaler` on training trajectories before the first batch (the "source scaler") and saves it inside the checkpoint under the key `source_scaler`. Note: the current pre-trained checkpoint may not include this key if it was trained before this feature was added; in that case, all scripts fall back to raw (un-normalized) data automatically.

---

### Step 3 - Train Baselines

These are independent and can be run in any order.

```bash
python -m baselines.train_gru_transfer       # GRU warm-start
python -m baselines.train_MAML               # MAML initialization
python -m baselines.train_transfer_weak      # Weak transfer (head-only, no encoder)
```

The scratch baseline needs no pre-training; it trains from random init per task at evaluation time.

---

### Step 4 - Adaptation and Evaluation

```bash
# Delete any stale results file first (column schema changed since last run)
rm -f results/gated_regularized_final.csv

python -m adaptation.gated_finetuning_regularized   # Model C (ours)
python -m baselines.adapt_scratch
python -m baselines.adapt_gru
python -m baselines.adapt_MAML
python -m baselines.adapt_transfer_weak
python -m baselines.adapt_persistence               # zero-parameter sanity-check baseline
```

Each script writes to `results/<name>_results.csv` and supports resuming from a partial run.

**Important: all baselines were audited for fairness before submission.** The following fixes were applied (see `methodology.txt` for full details):
- GRU now evaluates over the full 200-step rollout from `query[:, 0, :]`, not a 50-step window from mid-trajectory.
- Weak Transfer now uses 50 adaptation steps, matching all other baselines.
- Scratch now trains with path loss + head loss (not head-only), matching the supervision richness of Model C.

Any existing results CSVs from before these fixes are stale and should be deleted before re-running.

---

### Step 5 - Comparison Table and Hero Plot

```bash
python -m evaluation.final_comparison
python -m plots.plot_hero_curve
```

Outputs:
- `results/final_mse_comparison_table.csv` — human-readable table with cells formatted as **"mean ± std"** (standard deviation across tasks at each operating point)
- `results/final_mse_comparison_table_numeric.csv` — numeric mean-only table for downstream scripts
- `results/final_comparison_plot.png` — line plots with **95% bootstrap CI** shaded, one panel per regime

**Uncertainty reporting conventions (Phase 2 rubric):**
- Summary tables: `mean ± std` where std is the standard deviation across tasks within each (regime, steps) cell.  This quantifies per-task variability, not measurement noise.
- Hero plots: mean line with `±1.96 × SE` band (≈ 95% CI on the mean), computed per (method, steps_available) bucket across tasks.  Band edges are clamped to `mean × 1e-3` to stay positive on the log axis.
- `sns.lineplot(errorbar=("ci", 95))` is used in `evaluation/final_comparison.py`; `matplotlib.fill_between` with `±1.96 × SE` is used in `plots/plot_hero_curve.py` so that custom line styles and colors are fully controlled.

---

### Step 6 - Gate Calibration Plot

```bash
python -m evaluation.plot_gate_calibration
```

Reads from the already-generated results file. Outputs:
- `results/gate_calibration.png` - gate value vs. residual error, theoretical sigmoid overlaid, colored by regime
- `results/gate_calibration_by_steps.png` - same plot faceted by support steps

The gate formula is `g = sigmoid(20 * (0.05 - D_res))`. Because gate value is computed directly from residual error, the empirical points lie exactly on the sigmoid. The figure's purpose is to show where each regime sits on that curve: testA tasks cluster at low residual (gate opens, model trusts adaptation), testC tasks cluster at high residual (gate closes, falls back to prior).

---

### Step 7 - Gate Correction Study

```bash
python -m evaluation.gate_correction_study
```

Outputs:
- `results/gate_study_metrics.csv` — raw per-task results for all gate variants across all step counts
- `plots/gate_correction_analysis.png` — three-panel figure:
  - **Panel A**: Mean gate value ± 95% CI vs. support steps for V1/V2/V3 — shows V1 collapse and V2/V3 stability
  - **Panel B**: Gate value vs. raw residual D_res (scatter) — V1 clusters near zero for all tasks; V2/V3 spread across the sigmoid; theoretical V2 sigmoid overlaid at median data variance
  - **Panel C**: Grouped bar chart of MSE per regime at `steps=201` for all five modes (V1, V2, V3, always-on, always-off) on a log scale

**Three gate variants compared:**

| Variant | Formula | Description |
|---------|---------|-------------|
| V1 (Original) | `g = σ(α(τ − D_res))` | Raw residual; collapses to g≈0 when data variance is large |
| V2 (Normalized) | `g = σ(α(τ − D_res/(var+ε)))` | Dimensionless NMSE; τ=0.05 means "gate opens when model explains ≥95% of variance" |
| V3 (Adaptive τ) | `g = σ(α(τ(N) − NMSE))`, τ(N)=τ·√(N/N_max) | Stricter at short horizons; equals V2 at full horizon |

**Reference modes**: `always_on` (g=1, no fallback) and `always_off` (g=0, prior only) bound the performance range.

The script runs adaptation once per task and shares MC rollout samples across all five modes (efficiency: 2×MC instead of 5×MC per task via linearity of expectation).

---

### Step 8 - Ablation Study

```bash
python -m adaptation.ablation_gate_and_latent
```

Outputs:
- `results/ablation_gate_and_latent.csv`
- `results/ablation_summary.png`

**Gate ablation** (z_dim=16, always runs): compares three gate modes:

| gate_mode | Description |
|-----------|-------------|
| `adaptive` | Full safety gate, g = sigmoid(20*(0.05 - D_res)) - this is Model C |
| `always_on` | g fixed to 1; adapted model only, no fallback |
| `always_off` | g fixed to 0; safe prior only, adaptation ignored |

**Latent dimension sweep**: runs for whichever checkpoints exist:
```
checkpoints/meta_zdim8_epoch_50.pt    (z_dim=8)
checkpoints/meta_epoch_50.pt          (z_dim=16, default)
checkpoints/meta_zdim32_epoch_50.pt   (z_dim=32)
```

**You must generate the z_dim=8 and z_dim=32 checkpoints explicitly before running the
ablation script.** The script silently skips any missing checkpoints and will produce an
incomplete sweep if they are absent. Follow these steps in order:

```bash
# Step 8a — generate z_dim=8 checkpoint
# 1. Edit config/base_config.py: set   latent_dim = 8
python -m training.train_meta
mv checkpoints/meta_epoch_50.pt checkpoints/meta_zdim8_epoch_50.pt

# Step 8b — generate z_dim=32 checkpoint
# 2. Edit config/base_config.py: set   latent_dim = 32
python -m training.train_meta
mv checkpoints/meta_epoch_50.pt checkpoints/meta_zdim32_epoch_50.pt

# Step 8c — restore default and verify all three checkpoints exist
# 3. Edit config/base_config.py: set   latent_dim = 16
ls checkpoints/meta_zdim8_epoch_50.pt checkpoints/meta_epoch_50.pt checkpoints/meta_zdim32_epoch_50.pt
```

Once all three checkpoints are in place, run the ablation (a single run covers
both axes — gate ablation at z_dim=16 and the latent-dim sweep at z_dim ∈ {8,16,32}):

```bash
python -m adaptation.ablation_gate_and_latent
```

**Save outputs before any re-run.** The script appends to `results/ablation_gate_and_latent.csv`
and overwrites `results/ablation_summary.png`. Back up the files immediately after each
successful run so a restart does not clobber them:

```bash
# Run the ablation, then immediately archive the outputs with a timestamp
python -m adaptation.ablation_gate_and_latent
cp results/ablation_gate_and_latent.csv results/ablation_gate_and_latent_$(date +%Y%m%d_%H%M%S).csv
cp results/ablation_summary.png         results/ablation_summary_$(date +%Y%m%d_%H%M%S).png
```

The script supports resuming: if `results/ablation_gate_and_latent.csv` already exists it
skips completed (axis, z_dim, gate_mode, regime, theta_id, steps) combinations and only
appends new rows. To force a clean re-run, delete the CSV first.

---

### Step 9 - Manifold Distance Analysis (Section 4.2)

This quantifies the "continuous manifold" claim by measuring how far each test task lies from the training distribution in theta-space.

```bash
python evaluation/proximity_analysis.py
python evaluation/plot_proximity_synthetic.py
```

**Step 1** computes per-task L2 distances from each test task to its nearest training task, using the 110-dimensional flattened SDE parameter vector `[theta_b | theta_sigma]`. Writes `results/adaptation_with_distances.csv`.

**Step 2** generates `plots/manifold_distance_analysis.png`, a two-panel figure:
- RMSE vs. theta-space distance (positive trend validates the manifold hypothesis)
- Gate value vs. theta-space distance (negative trend shows the gate automatically reduces trust as novelty increases)

Both panels show the Pearson r value in the corner. If `results/maml_results_full.csv` or
`results/transfer_weak_results_full.csv` are present, MAML and Transfer points are overlaid
in the RMSE panel for direct comparison.

Expected results (verified 2026-04-19):
```
Regime A   avg distance = 2.68   avg RMSE = 0.379   avg gate = 0.667
Regime B   avg distance = 4.18   avg RMSE = 0.702   avg gate = 0.603
Regime C   avg distance = 4.97   avg RMSE = 1.002   avg gate = 0.491

Pearson r(distance, RMSE) = 0.90
```

---

### Step 10 - Regime-Switch Experiment

```bash
python -m evaluation.regime_switch_experiment
```

Simulates a sudden physics-regime change (testA → testC) midway through an online inference sequence and measures how quickly the model adapts its latent representation.

**Protocol**: For each of N_SEEDS=5 independent (taskA, taskC) task pairs, a "hard shock" ground-truth sequence is constructed by concatenating T_PRE=50 steps of a testA trajectory with T_POST=100 steps of a testC trajectory. At each time step the model re-encodes the most recent CONTEXT_LEN=20 observations via a sliding window, makes a one-step Euler prediction, and the error is recorded.

**Metrics computed across seeds (mean ± std):**

| Metric | Definition |
|--------|-----------|
| `peak_shock_error` | Maximum per-step MSE in the post-shock window (steps T_PRE … T_PRE + T_POST) |
| `recovery_steps` | First post-shock step where per-step MSE ≤ pre-shock baseline mean; set to T_POST if MSE never returns to baseline |

**Outputs:**
- `results/regime_switch_metrics.csv` — per-seed values of baseline_mse, peak_shock_error, recovery_steps
- `plots/regime_switching_analysis.png` — two-panel figure:
  - **Panel A** (Error Trajectory): mean MSE ± 1 std across seeds; vertical dashed line at the shock; recovery window shaded green (orange if no recovery); horizontal dotted line at pre-shock baseline; peak shock annotated with mean ± std
  - **Panel B** (Latent Norm Trajectory): mean ||z_t||₂ ± 1 std across seeds; same shock line and recovery shading — shows whether and when the latent representation shifts to track the new regime

---

### Optional Sweeps

```bash
python -m baselines.sweep_shots       # N_SHOTS in {2,3,4,5,8,10}
python -m baselines.sweep_iterations  # adaptation step budget sweep
```

---

## Phase 3: Sensitivity Analyses

These scripts produce appendix figures that stress-test the robustness of
the main results.  They can be run independently of each other and in any
order, but each requires the primary checkpoint (`checkpoints/meta_epoch_50.pt`)
and the test-split data to be present.

---

### Phase 3, Step 1 — Adaptation-Budget Sensitivity Sweep

```bash
python -m evaluation.sweep_adaptation_budget
```

Evaluates Model C across four test-time optimisation budgets:
`adapt_steps ∈ {10, 25, 50, 100}`.  The main evaluation fixes `ADAPT_STEPS=50`;
this sweep confirms that the performance advantage is not a fragile artefact
of that single setting.

Each budget is evaluated at three context lengths (`steps_available ∈ {20, 50, 201}`)
and across all three test regimes (`testA`, `testB`, `testC`), so the full grid
is 4 × 3 × 3 = 36 (budget, context, regime) cells, each aggregated over all 30
tasks in the regime.

**Outputs:**
- `results/adaptation_budget_metrics.csv` — raw per-task metrics
  (columns: `adapt_budget`, `steps_available`, `regime`, `theta_id`,
  `gate_value`, `residual_error`, `adapt_time`, `mse_rollout`, `mse_final`,
  `mse_1step`, `rmse_rollout`, `rmse_final`, `rmse_per_dim_mean`,
  `rmse_per_dim_max`, `nll`)
- `plots/adaptation_budget_sensitivity.png` — three-panel appendix figure
  (one panel per regime, x-axis = adapt\_steps, y-axis = rollout MSE,
  one line per context length with ±95% CI bands, vertical dotted line
  marking the paper's default of 50 steps)

**Expected result:** MSE should improve quickly from 10→25→50 steps and then
plateau or improve only marginally at 100 steps.  The relative ordering of
regimes (testA < testB < testC in MSE) should be preserved across all budgets.
If it is, the main-paper result at 50 steps is representative, not cherry-picked.

**Resume:** the script appends to the CSV and skips already-evaluated rows.
To force a clean re-run, delete `results/adaptation_budget_metrics.csv` first.

---

### Phase 3, Step 2 — Manifold Distance Analysis (already in Step 9)

```bash
python evaluation/proximity_analysis.py
python evaluation/plot_proximity_synthetic.py
```

See Step 9 above.  The plot is now saved to `plots/manifold_distance_analysis.png`.

---

## Part 2: DeepMind Control Experiments

These require `dm_control` and are heavier than the synthetic experiments.

### Step 1 - Generate DM Data

```bash
python -m deepmind.generate_all
```

Generates Reacher, Finger, and Cheetah data under three physics regimes (testA: in-distribution; testB: mild extrapolation; testC: heavy OOD). Output: `data/deepmind/{reacher,finger,cheetah}/index.csv`.

### Step 2 - Train Models

```bash
python -m deepmind.train_manager
```

Trains one meta-model per task sequentially. Checkpoints saved to `checkpoints/deepmind/{task}/model_best.pt`.

### Step 3 - Evaluate

```bash
python -m deepmind.benchmark_manager   # runs all three tasks in sequence
```

Or to run a single task, edit `TARGET_DATASET` at the top of `deepmind/gated_finetuning_regularized_dm.py` and run it directly. Each run saves to `results/dm_{task}_model_c.csv`.

### Step 4 - Plot Results

```bash
python -m deepmind.plot_benchmark_results
```

**Warning:** `plot_benchmark_results.py` contains hardcoded result values for Finger and Cheetah. Those hardcoded values are from an earlier run and are suspected to reflect dimensional collapse (MSE values near 1e-17 to 1e-23 are physically impossible in float32 and likely mean the model is predicting near-constant trajectories). After re-running Step 3 for those tasks, replace the hardcoded dicts with values from `results/dm_finger_model_c.csv` and `results/dm_cheetah_model_c.csv`. Use per-dimension RMSE to verify the model is actually predicting meaningfully across all dimensions.

---

## Key Output Files

| File | Script | Used in |
|------|--------|---------|
| `results/gated_regularized_final.csv` | `adaptation/gated_finetuning_regularized.py` | comparison table |
| `results/gru_baseline_sweep.csv` | `baselines/adapt_gru.py` | comparison table |
| `results/maml_results_full.csv` | `baselines/adapt_MAML.py` | comparison table |
| `results/transfer_weak_results_full.csv` | `baselines/adapt_transfer_weak.py` | comparison table |
| `results/scratch_sweep_results_full.csv` | `baselines/adapt_scratch.py` | comparison table |
| `results/persistence_results_full.csv` | `baselines/adapt_persistence.py` | sanity-check lower bound |
| `results/final_mse_comparison_table.csv` | `evaluation/final_comparison.py` | paper appendix |
| `plots/manifold_distance_analysis.png` | `evaluation/plot_proximity_synthetic.py` | paper Figure (Section 4.2) |
| `results/gate_calibration.png` | `evaluation/plot_gate_calibration.py` | paper Figure (Gate) |
| `results/gate_study_metrics.csv` | `evaluation/gate_correction_study.py` | gate variant comparison |
| `plots/gate_correction_analysis.png` | `evaluation/gate_correction_study.py` | paper Figure (Gate Study) |
| `results/regime_switch_metrics.csv` | `evaluation/regime_switch_experiment.py` | paper Figure (Regime Switch) |
| `plots/regime_switching_analysis.png` | `evaluation/regime_switch_experiment.py` | paper Figure (Regime Switch) |
| `results/ablation_gate_and_latent.csv` | `adaptation/ablation_gate_and_latent.py` | paper Table (Ablation) |
| `results/adaptation_budget_metrics.csv` | `evaluation/sweep_adaptation_budget.py` | paper Appendix (Budget Sensitivity) |
| `plots/adaptation_budget_sensitivity.png` | `evaluation/sweep_adaptation_budget.py` | paper Appendix (Budget Sensitivity) |
| `results/dm_{task}_model_c.csv` | `deepmind/gated_finetuning_regularized_dm.py` | DM results table |

---

## Uncertainty Reporting

All summary tables report **mean ± std** and all hero plots render **95% confidence intervals**.  See Step 5 above for the precise definitions.

---

## Metrics

All evaluation scripts report the following:

- `mse_rollout`: MSE over the full trajectory (batch x time x dims)
- `mse_final`: MSE at the terminal timestep only
- `mse_1step`: MSE strictly at the first predicted timestep (t=1); diagnostic tool — a model that cannot beat the persistence baseline on `mse_1step` is not learning useful short-horizon dynamics
- `nll`: Gaussian NLL using MC sample variance
- `rmse_per_dim_max`: worst-case single-dimension RMSE (diagnostic for dimensional collapse; if this is more than 10x `mse_rollout^0.5`, the model is ignoring some dimensions)
- `gate_value`: safety gate g in [0,1]; values near 1 mean the model trusts the adapted prediction, values near 0 fall back to the prior
- `residual_error`: support-set MSE after adaptation, which feeds the gate formula

---

## Config Reference

All hyperparameters live in `config/base_config.py`. Key values you might need to adjust:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `x_dim` | 10 | State dimension (synthetic). Auto-overridden for DM tasks. |
| `latent_dim` | 16 | Latent code z dimension |
| `n_steps` | 200 | SDE integration steps per trajectory |
| `n_train_thetas` | 150 | Distinct SDE parameter sets in training |
| `ADAPT_STEPS` | 50 | Gradient steps at adaptation time |
| `N_SHOTS` | 2 | Support trajectories per task |
| `BETA_REG` | 0.01 | L2 regularization weight on z |
| `GATE_ALPHA` | 20.0 | Gate sharpness parameter |
| `GATE_TAU` | 0.05 | Gate threshold (residual error at which g = 0.5) |

Do not manually change `x_dim` for DM experiments; `train_manager.py` and `benchmark_manager.py` detect it automatically.
