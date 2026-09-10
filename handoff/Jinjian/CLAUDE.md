# Agent instructions — Jinjian's package (Measurement)

You are working inside the Meta-Diffusion repository on branch `jinjian`. Read
`handoff/Jinjian/README.md` for the scientific brief and `docs/PROTOCOL_CARD.md` for the
values you may not change. This file states the rules that must not be broken.

## Environment

- Windows, PowerShell 5.1. `&&` is a parser error — one command per line.
- Interpreter: `.venv/Scripts/python.exe`. Python 3.12, torch with CUDA.
- Set `PYTHONIOENCODING=utf-8` for anything that prints non-ASCII.
- Run everything from the repository root.
- CIFAR-100 is already at the relative default; do not set `CIFAR100_ROOT`.

## Files you may create or modify

```
scripts/posthoc_controls.py      (new)
scripts/gen_specificity.py       (new)
evaluation/instruments.py        (new)
scripts/cifar_table.py
scripts/phase_d_sweeps.sh
handoff/Jinjian/results/*
```

## Files you must not touch, for any reason

```
models/    training/    diagnostics/    config/    episodes/    adaptation/
artifacts/*.json        any existing checkpoint
```

Another person owns each of those. If you find a bug in one, write it down in your
results file and carry on. Do not fix it, and do not work around it by copying the file.

Specifically: **do not import `diagnostics/controls.py`.** Write your own control loop.
The duplication is intentional — it is a cross-check on a quantity this project has
already misreported twice.

## Rules for measurement code

1. **Rebuild from the checkpoint's stored config**, never from `BaseConfig()` defaults.
   `scripts/cifar_table.py` lines 84-96 show the pattern. A model rebuilt from live
   defaults crashes loudly when dimensions differ and is silently wrong when they match.
2. **Every control shares one noising draw.** Use `ScoreModel.eps_hat_many`, which takes
   several coordinates against one `(x_t, t, eps)`. Comparing controls across separate
   `q_sample` calls throws away a factor of sixty-five in variance.
3. **Report absolute paired differences with intervals**, plus the step count. Never
   report `task_specific_frac` as a headline: its denominator decays to zero and changes
   sign.
4. **Never let the coordinate and the reference share images.** Split the target query
   pool in half.
5. **A check that can fail silently is not a check.** Write verifications as Python
   assertions that raise, not as greps that can return empty.

## Checkpoint and split pairing

| Checkpoint | Split file to pass with `--domainshift-path` |
|---|---|
| `cifar_ds3_step50000.pt` | `artifacts/cifar100_domainshift_c3.json` |
| `cifar_ds_blur_step50000.pt` | `artifacts/cifar100_domainshift.json` |
| `ds_cap32_60k_step60000.pt` | `artifacts/cifar100_domainshift.json` |

Mixing them trips the corruption-count guard. The guard is correct; do not weaken it.

## Order of work

1. `bash handoff/Jinjian/verify.sh` — preflight.
2. `scripts/posthoc_controls.py`, then run it on both 128-channel checkpoints.
3. `evaluation/instruments.py`, then `scripts/gen_specificity.py`.
4. The two sweep-harness edits.
5. Copy results into `handoff/Jinjian/results/`.

`bash handoff/Jinjian/run_all.sh` runs steps 2 to 4 once the scripts exist.

## What you must not do

- Do not train anything.
- Do not draw a conclusion about whether the method works. Deliver numbers; the decision
  happens at integration against criteria fixed in advance.
- Do not add a dependency (no torchvision, no Inception, no torchmetrics) without saying
  so explicitly in the results file. The repository runs on torch and numpy alone.
- Do not commit anything under `artifacts/` — it is ignored for text and images.
- Do not push to a branch that is not `jinjian`.
