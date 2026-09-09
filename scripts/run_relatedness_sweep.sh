#!/usr/bin/env bash
# Stage 1: the relatedness sweep, which measures how much weight the source
# document's qualifier 'within a related family of tasks' actually carries.
#
# perturb 0 -> every task identical; perturb 1 -> back to mutually unrelated.
set -u
. "$(dirname "$0")/_common.sh"

for p in 0.1 0.25 0.5 1.0; do
  echo "### related@$p"
  $PY -m runner.stage1_gmm --steps 80000 --k 16 --decoder linear \
      --family related --perturb $p --eval-tasks 4 \
      > artifacts/stage1_related_p${p}.log 2>&1
done
echo "### unrelated control (re-run so the checkpoint metadata matches the others)"
$PY -m runner.stage1_gmm --steps 80000 --k 16 --decoder linear \
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
