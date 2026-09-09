#!/usr/bin/env bash
# Stage 1: compare k = 32 against k = 64 at an identical 80k steps, which removes
# the confound that a larger coordinate is simply harder to train.
set -u
. "$(dirname "$0")/_common.sh"

for k in 32 64; do
  echo "### k=$k steps=80000"
  $PY -m runner.stage1_gmm --steps 80000 --k $k --decoder linear \
      --family unrelated --eval-tasks 4 > artifacts/stage1_k${k}_s80k.log 2>&1
done
$PY scripts/ft_reference.py stage1_linear_k16_s80000_unrelated.pt \
    stage1_linear_k32_s80000_unrelated.pt stage1_linear_k64_s80000_unrelated.pt \
    > artifacts/ksweep_80k_reference.txt 2>&1
echo "### done"
