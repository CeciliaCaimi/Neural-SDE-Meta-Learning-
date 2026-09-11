# handoff/

Three people are running the next stage of this project in parallel, on three different
machines, **without coordinating with each other**. This directory is the hand-out: one
folder per person, containing the instructions, the driver scripts, and the place to put
results.

Everything the three share — fixed hyperparameters, split-file checksums, reporting
rules, file ownership — is in [`docs/PROTOCOL_CARD.md`](../docs/PROTOCOL_CARD.md). Read
that first. Nothing in it is negotiable locally; if it looks wrong, raise it with the
person who handed out the work rather than changing it in your branch.

Written in English, like the rest of the repository.

---

## Who runs what, and why

| | Machine | Package | Why this machine |
|---|---|---|---|
| **Jinjian** | local workstation, RTX 5080 (16 GB) | **Measurement** — E12a post-hoc controls, E12 generation-level test, E14 sweep harness | The three trained checkpoints are ~430 MB each, are not in git, and exist only on this machine. CIFAR-100 also lives here. Sampling 20 480 images needs a CUDA GPU, but only for minutes. |
| **Jing Peng** | AWS GPU server | **Training** — E13 centred coordinates (4 runs), E15 generic-latent comparison arm | 5.5 GPU-hours of training, and the only machine that can run several of them at once. Everything needed travels in the clone except CIFAR-100, which `setup.sh` fetches. |
| **Yu Cao** | Mac, no CUDA | **Positive control** — the stage-1 four-way panel and the task-distance sweep | Stage 1 is two-dimensional Gaussian mixtures: synthetic, so no dataset, and measured at **54 iterations per second on CPU** — 80 000 steps in 24 minutes. It needs no GPU and no data transfer, and it supplies the one number the CIFAR arm cannot: what the effect looks like where the mechanism demonstrably works. |

The allocation follows the hardware, not seniority. The person with the data and the
checkpoints measures; the person with the compute trains; the person with a laptop runs
the domain that fits on a laptop and happens to be the scientifically decisive control.

## The three devices that make "no coordination" work

1. **A shared baseline, fixed before the split.** The frozen task family
   `artifacts/cifar100_domainshift_c3.json`, a `model_cls` seam in `training/loop.py`, a
   `--domainshift-path` flag on the runners, and `CIFAR100_ROOT` for machines whose data
   is not where this one keeps it. All of it is already in the commit you cloned.
2. **One owner per file.** The table in the protocol card. Check your column before
   editing anything; if a path is not yours, do not touch it even to fix an obvious bug —
   note it in your results file instead.
3. **No one draws the conclusion.** Deliver numbers. Whether the generation test
   separates the coordinates, and whether the task-specific effect clears zero, is decided
   at integration against criteria that are written down *before* anyone sees a result.

## Branches, and how to clone

| Person | Branch |
|---|---|
| Jinjian | `jinjian` |
| Jing Peng | `jing-peng` |
| Yu Cao | `yu-cao` |

**Clone your branch by name.** A plain `git clone` checks out this repository's default
branch, which is `main` and is *not* this project — you would get a tree with no
`handoff/` directory and wonder where the work went.

```bash
git clone -b jinjian https://github.com/CeciliaCaimi/Neural-SDE-Meta-Learning-.git
```

The three branches and `meta-diffusion-clean` all point at the same commit today. Work on
yours, push to yours, and do not rebase onto anyone else's.

## Where results go

**Not** in `artifacts/`. That directory ignores `*.txt`, `*.png`, `*.out` and the whole
of `checkpoints/`, so anything you put there will silently fail to commit. Write
deliverables to `handoff/<YourName>/results/`, which is tracked.

Training logs are the exception worth naming: `checkpoints/<run>_log.jsonl` is ignored,
so copy the log into your results directory when you are done. It is a few hundred
kilobytes and it is the primary evidence for two of the four deliverables.

## What is in each folder

| File | Purpose |
|---|---|
| `README.md` | Your package: what to build, what to run, what to deliver, how to know you are finished |
| `CLAUDE.md` | The same thing addressed to a coding agent, with the constraints stated as rules it must not break |
| `verify.sh` | Preflight. Run it first; it checks the interpreter, the device, the split checksum and the test suite |
| `setup.sh` | Environment bootstrap (Jing Peng and Yu Cao only — Jinjian's machine is already set up) |
| `run_all.sh` | The driver: the exact sequence of commands for your package |
| `results/` | Where your deliverables go |
