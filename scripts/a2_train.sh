#!/usr/bin/env bash
# A2: train the two conditioning arms back to back on the one GPU.
#
# Both arms take the same flags. They differ only in the entry point, which swaps the
# backbone for film_unet and the composition for FiLMScoreModel; encoder, transport,
# relation descriptor, episode construction, split, schedule, objective, optimiser, budget
# and seed all come from the same code path.
#
# The basis arm is retrained rather than reusing cifar_ds3_step50000, whose stored config
# has drifted from current defaults (adapt.noise_batch 16 -> 32, diagnose_every 1000 -> 500,
# ckpt_every 10000 -> 2000). Diagnostics draw from the global RNG, so a different cadence
# is a different training trajectory: matching the arms mattered more than saving 75
# minutes. cifar_ds3 is still reported alongside, as the checkpoint A1 and A3 were measured
# on.
#
#     bash scripts/a2_train.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SPLIT=artifacts/cifar100_domainshift_c3.json
OUT=handoff/results
export PYTHONIOENCODING=utf-8

COMMON="--scheme domainshift --domainshift-path $SPLIT --k 16 --steps 50000
        --diagnose-every 2000 --ckpt-every 10000"

mkdir -p "$OUT"

echo "=== A2 basis arm  $(date) ==="
$PY -m runner.train $COMMON --run-name a2_basis > "$OUT/A2_train_basis.log" 2>&1
echo "basis exit $?"

echo "=== A2 FiLM arm (per_block)  $(date) ==="
$PY -m runner.train_film $COMMON --film-mode per_block --run-name a2_film \
    > "$OUT/A2_train_film.log" 2>&1
echo "film exit $?"

echo "=== done  $(date) ==="
ls -la checkpoints/a2_basis_step50000.pt checkpoints/a2_film_step50000.pt
