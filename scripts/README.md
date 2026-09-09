# scripts/

Analysis and sweep drivers. Nothing here is imported by the framework — these produce
the tables, figures and diagnostics that the report quotes, and every one of them can
be deleted without affecting training.

All shell scripts source [`_common.sh`](_common.sh), which moves to the repository
root, picks the interpreter (the project virtual environment when present, otherwise
`python` on PATH) and forces UTF-8 output. So they run identically from anywhere:

```bash
bash scripts/sweep_k.sh
```

Python scripts take the repository root as the working directory:

```bash
python scripts/cifar_table.py checkpoints/<run>_step50000.pt --split val
```

---

## Evaluation and diagnostics

These are the ones worth reaching for. Each answers a specific question, named below.

| Script | Question it answers | Report |
|---|---|---|
| `cifar_table.py` | How does every adaptation strategy compare on held-out episodes, with paired confidence intervals? | E4, E8 |
| `probe_transport.py` | Does transport move `z_S` toward a coordinate encoded from abundant target data? Separates "the coordinate is wrong" from "the basis cannot use it". | E11 |
| `probe_collapse.py` | At which layer does the encoder's coordinate collapse to a near-constant? | E2 |
| `paired_ci.py` | Stage-1 baselines with paired confidence intervals over tasks and seeds. | E1 |
| `final_table.py` | The stage-1 main comparison table. | E1 |
| `ft_reference.py` | The upper bound: how far does unconstrained full fine-tuning get? Taken as the minimum over a budget grid, because fine-tuning overfits a finite target sample. | E1 |
| `stage1_missing_metrics.py` | Two quantities the source specification requires that the stage-1 runner did not compute. | E1 |

## Figures

| Script | Output |
|---|---|
| `gen_tree_tex.py` | The three-way split tree, as TikZ, generated from `artifacts/cifar100_domainshift.json` so the figure cannot drift from the split the code actually loads |
| `make_gmm_figure.py` | The stage-1 mixture scatter, as inline SVG |
| `export_gmm_pdf.py` | The same figure as vector PDF, for `\includegraphics` |

## CIFAR-100 sweeps

Run in this order; each depends on what the previous one settled.

| Script | What it varies | Finding |
|---|---|---|
| `sweep_k.sh` | Phase A — coordinate dimension `k` in {16, 32, 64}, on validation classes | `k` is not the bottleneck; all three decay to nothing |
| `phase_b_capacity.sh` | Phase B — backbone width at 128 / 64 / 32 channels, `k` held at 32 | Apparent gain rises 725-fold as the backbone shrinks |
| `phase_c_converge.sh` | Phase C — the two small backbones run to 60k steps | The decay is universal; capacity buys time, not permanence |
| `phase_d_sweeps.sh` | Phase D — `K_T` in {1,2,5,10,20} and `M_S` in {16 … 300}, evaluation only | Transport's margin behaves as predicted, but every strategy lands on the same loss |

**Phase B and C were read as a success at the time, and that reading was wrong.**
`gain_vs_zero` measures whether the model needs a non-zero coordinate, not whether it
uses the coordinate to identify the task. The mean-coordinate substitution in
`diagnostics/controls.py` separates the two and reports `task_specific_frac`; read
that before `gain_vs_zero`. The story is in the root [`README.md`](../README.md).

## Stage-1 (Gaussian mixture) sweeps

Historical. They reproduce the stage-1 results, but each expects checkpoints from the
runs before it, named in the script. Read them as a record of what was run rather than
as a pipeline to execute from scratch.

| Script | What it runs |
|---|---|
| `run_stage1_suite.sh` | Linear against non-linear coordinate decoder, then the `k` sweep at 40k steps |
| `k_matched.sh` | `k` = 32 against 64 at a matched 80k steps, removing the "larger is harder to train" confound |
| `k_low_matched.sh` | The low end, `k` = 2, 4, 8, also at 80k |
| `k_convergence.sh` | `k` = 32 and 64 to 160k steps — is the slowdown saturation or undertraining? |
| `k16_converge.sh`, `k64_converge.sh` | Both to 320k steps, where the captured fraction levels off |
| `run_relatedness_sweep.sh` | Task-family relatedness at perturbation 0.1 … 1.0 |
| `rerun_reference.sh` | Rescores existing checkpoints under the minimum-over-budget-grid rule. No training |

Their console logs are in [`../artifacts/`](../artifacts/).
