# Jing Peng results (training package)

Two experiments: E13 (centred coordinates) and E15 (additive basis vs generic FiLM).

## E13 - centred coordinates
Source: run `jp-all-20260912-173432` (original 5-run job).
- `E13_panel.txt`
- `ctr32_off_log.jsonl`, `ctr32_on_log.jsonl`      (k=32, base_channels=32, 60k steps)
- `ctr128_off_log.jsonl`, `ctr128_on_log.jsonl`    (k=16, base_channels=128, 50k steps)
- matching `*_console.txt`

## E15 - generic FiLM vs additive basis (spec-correct PER-BLOCK FiLM)
Source: re-run `jp-genpair-20260918-130056` (basis + per-block FiLM trained together, 50k steps).
- `E15_film_panel.txt`
- `film128_log.jsonl` / `film128_console.txt`      (per-block FiLM)
- `e15_basis_log.jsonl` / `e15_basis_console.txt`  (matched additive basis for the comparison)

`verify_console.txt` - GPU preflight (all 5 test modules pass, incl. test_cifar100_episodes 10/10).

## Checkpoints (NOT in git - 430 MB each)
Final step-50000 checkpoints for the E15 generation-level comparison are in S3:
```
s3://neural-sde-meta-learning-103631805002/jp/checkpoints/jp-genpair-20260918-130056/
  ctr128_off_step50000.pt   (additive basis)
  film128_step50000.pt      (per-block FiLM)
```

## Notes
- Denoising `delta_task`/loss panels are supporting diagnostics. The decisive E15 result is
  generation-level (K_T in {1,5,20}), run with the S3 checkpoints via the measurement package's evaluator.
- The original run's `film128` used a weaker timestep-embedding FiLM; it is superseded here by
  the per-block FiLM (matches the FiLM implementation note).
