#!/usr/bin/env bash
# Stage 1: the relatedness sweep, which measures how much weight the source
# document's qualifier 'within a related family of tasks' actually carries.
#
# perturb 0 -> every task identical; perturb 1 -> back to mutually unrelated.
#
# Step count: 80000 by default, overridable for a slower machine. Whatever it is, it must
# be the same across all five points -- comparing them is the whole purpose, and a
# consistent step count matters far more than a large one.
#
#     STAGE1_STEPS=40000 bash scripts/run_relatedness_sweep.sh
set -u
. "$(dirname "$0")/_common.sh"

STEPS="${STAGE1_STEPS:-80000}"
echo "### steps per point: $STEPS"

for p in 0.1 0.25 0.5 1.0; do
  echo "### related@$p"
  $PY -m runner.stage1_gmm --steps "$STEPS" --k 16 --decoder linear \
      --family related --perturb $p --eval-tasks 4 \
      > artifacts/stage1_related_p${p}.log 2>&1
done
echo "### unrelated control (re-run so the checkpoint metadata matches the others)"
$PY -m runner.stage1_gmm --steps "$STEPS" --k 16 --decoder linear \
    --family unrelated --eval-tasks 4 \
    > artifacts/stage1_related_unrelated.log 2>&1

echo "### upper-bound reference"
$PY scripts/ft_reference.py \
    stage1_linear_k16_s80000_related0.1.pt \
    stage1_linear_k16_s80000_related0.25.pt \
    stage1_linear_k16_s80000_related0.5.pt \
    stage1_linear_k16_s80000_related1.0.pt \
    stage1_linear_k16_s80000_unrelated.pt \
    > artifacts/relatedness_reference.txt 2>&1
echo "### done"
