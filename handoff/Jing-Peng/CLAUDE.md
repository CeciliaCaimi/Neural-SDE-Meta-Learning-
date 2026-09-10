# Agent instructions — Jing Peng's package (Training)

You are working inside the Meta-Diffusion repository on branch `jing-peng`, on an AWS GPU
server. Read `handoff/Jing-Peng/README.md` for the scientific brief and
`docs/PROTOCOL_CARD.md` for the values you may not change. This file states the rules
that must not be broken.

## Environment

- Linux. Run `bash handoff/Jing-Peng/setup.sh` once before anything else.
- Interpreter: `.venv/bin/python`. The shell drivers pick it up automatically.
- CIFAR-100 is **not** in the repository and must never be added to it. `setup.sh`
  downloads it and sets `CIFAR100_ROOT`; export that variable in every shell you use.
- Run everything from the repository root.
- Pass `--domainshift-path artifacts/cifar100_domainshift_c3.json` on every run. All new
  training uses the three-corruption family.

## Files you may create or modify

```
models/score_model.py        training/loop.py         training/meta_train.py
diagnostics/controls.py      config/base_config.py    runner/train.py
tests/test_score_identity.py
models/film_unet.py          (new)
baselines/film_conditioning.py (new)
runner/train_film.py         (new)
handoff/Jing-Peng/results/*
```

## Files you must not touch, for any reason

```
scripts/    evaluation/    episodes/    adaptation/    domains/
artifacts/*.json           handoff/Jinjian/    handoff/Yu-Cao/
```

Another person owns each. If you find a bug in one, write it down in your results file
and carry on.

`runner/train_film.py` must use the seam that already exists — `build()` and `train()`
take `model_cls`. **Do not edit `training/loop.py` to add the FiLM arm**, even though you
own that file: the seam is there precisely so the comparison arm and the centring change
stay separable, and mixing them makes the matched comparison unverifiable.

## Hard rules for the centring change

1. **`z_center` is a buffer with no gradient path.** Subtract it inside `_prepare_z`;
   never let a gradient reach it.
2. **Exclude `z_center` from the EMA shadow.** `EMA.__init__` in `training/loop.py`
   shadows every floating-point state-dict entry. It drifts, so shadowing it makes the
   diagnostics read a centre training never used.
3. **Do not delete the identity assertion** in `tests/test_score_identity.py`.
   Parameterise it: with centring off, `eps_hat(0) == eps_hat_0`; with centring on,
   `eps_hat(z̄) == eps_hat_0`. Bit for bit, in both cases.
4. **Drop `loss_zero` from the centred panel.** Under centring it is not a null and will
   be misread.
5. **Name the new log fields exactly `delta_task` and `z_center_norm`.** Someone else's
   plotting script consumes them and you cannot ask them to change it.
6. **Add no new network.** The supervisor's note rules out a wider transport, confidence
   gates, separate encoders, and probabilistic latent machinery. The change is a buffer
   and a subtraction.

## Rules for reporting

- Read nothing before the curve has levelled: quote the mean over the **final fifth** of
  training and the slope over the last 10 000 steps, and always print the step count.
- Report absolute paired differences with 95 % intervals. Never headline
  `task_specific_frac`; its denominator decays to zero and changes sign.
- All controls share one noising draw — keep using `eps_hat_many`.
- All diagnostics on the **validation** split. `test` is untouched.

## Order of work

1. `bash handoff/Jing-Peng/setup.sh` then `bash handoff/Jing-Peng/verify.sh`.
2. The four centring edits, then the whole test suite with the flag both ways.
3. `ctr32_off` and `ctr32_on` — under an hour together. Look at the sign of `delta_task`.
4. `ctr128_off` and `ctr128_on` regardless of that sign: positive confirms it on the
   protocol backbone, negative rules out a small-backbone artefact.
5. The FiLM arm: code, `check_backbone`, then `film128`.
6. Copy results and the `*_log.jsonl` files into `handoff/Jing-Peng/results/`.

`bash handoff/Jing-Peng/run_all.sh` runs steps 3 to 5. If your instance has several GPUs,
the runs are independent — set `CUDA_VISIBLE_DEVICES` per run and start them in parallel.

## What you must not do

- Do not commit a checkpoint. They are 430 MB and `checkpoints/` is ignored; keep them on
  the server and say where they are.
- Do not commit anything under `artifacts/` — text and images there are ignored.
- Do not decide whether the method works. Deliver numbers.
- Do not change any value in `docs/PROTOCOL_CARD.md`. The comparison between your two
  arms and the other packages is only meaningful because those values are identical
  everywhere.
- Do not push to a branch that is not `jing-peng`.
