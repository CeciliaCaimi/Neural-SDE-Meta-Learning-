# Repository instructions

Meta-Diffusion: few-shot adaptation of a diffusion model by fitting a `k`-dimensional task
coordinate against a frozen network. `README.md` is the overview, `ALGORITHM.md` is the
full specification, and `docs/meta_diffusion_combined.pdf` is the 28-page record of
experiments E1–E11.

## If you are working on the next stage

The work is split three ways across three machines, and the three packages must not
depend on each other. **Find your branch in this table and read both files before writing
any code.**

| Branch | Package | Binding instructions |
|---|---|---|
| `jinjian` | Measurement — generation-level task specificity, post-hoc controls, sweep harness | `handoff/Jinjian/CLAUDE.md`, `handoff/Jinjian/README.md` |
| `jing-peng` | Training — centred coordinates, the four-way panel, the generic-latent arm | `handoff/Jing-Peng/CLAUDE.md`, `handoff/Jing-Peng/README.md` |
| `yu-cao` | Positive control — the stage-1 panel and the task-distance sweep | `handoff/Yu-Cao/CLAUDE.md`, `handoff/Yu-Cao/README.md` |

Clone by branch name — the repository's default branch is `main`, which is a different
project and has no `handoff/` directory:

```bash
git clone -b <your-branch> https://github.com/CeciliaCaimi/Neural-SDE-Meta-Learning-.git
```

`docs/PROTOCOL_CARD.md` fixes everything the three share: hyperparameters, split-file
checksums, reporting rules, and one owner per file. Nothing in it may be changed locally.
`docs/work_split.pdf` is the same allocation keyed to the supervisor's numbered note.

Run `bash handoff/<YourName>/verify.sh` before you start. It checks the interpreter, the
device, the dataset, the split checksums and the test suite, and it is faster than
finding out later.

## Rules that hold on every branch

- **One owner per file.** If a path is not in your package's list, do not edit it — even
  to fix an obvious bug. Record the bug in your results file instead.
- **Deliver numbers, not conclusions.** Whether the method works is decided at
  integration, against criteria written down before any result was seen.
- **Read nothing before the curve has levelled.** Quote the mean over the final fifth of
  training, and state the step count beside every number.
- **Report absolute paired differences with 95 % intervals.** Never headline
  `task_specific_frac`: its denominator decays towards zero and changes sign.
- **All controls share one noising draw** (`ScoreModel.eps_hat_many`). Between-episode
  variance is about sixteen times the effect; pairing reduces the standard deviation by a
  factor of sixty-five.
- **Rebuild evaluation objects from the checkpoint's own stored config**, never from
  current defaults.
- **Results go in `handoff/<YourName>/results/`.** `artifacts/` ignores `*.txt`, `*.png`,
  `*.out` and all of `checkpoints/`, so files put there vanish without an error.
- **A verification that can fail silently is not a verification.** Write checks as Python
  assertions that raise, not as greps that can return empty.

## Data

CIFAR-100 is never committed and never copied into the working tree. Point at your own
copy with `CIFAR100_ROOT`, set before the process starts. The directory holding the
dataset on the machine of record also contains clinical data and a personal identity
document, so nothing above the repository root may be added here. Stage 1 needs no
dataset at all — its Gaussian mixtures are generated from a seed.

## Environment

Python 3.11+. The project virtual environment is `.venv/`, picked up automatically by the
shell drivers through `scripts/_common.sh`. On Windows the interpreter is
`.venv/Scripts/python.exe`, PowerShell 5.1 does not accept `&&`, and any script printing
non-ASCII needs `PYTHONIOENCODING=utf-8`.

Verify a change with the test suite — 72 assertions, most of which need no GPU:

```bash
python -m tests.test_score_identity
python -m tests.test_backbone_swap
python -m tests.test_gmm_analytic
python -m tests.test_cifar100_episodes
python -m tests.test_document_conformance
```

`test_cifar100_episodes` is the only one that loads the dataset.
