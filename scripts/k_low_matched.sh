#!/usr/bin/env bash
# Stage 1: the low end of the k sweep, k = 2, 4, 8, all at a matched 80k steps.
#
# Together with k_matched.sh this gives one step-matched curve across the whole
# range k = 2 ... 64, which is the only form in which the sweep can be read.
set -u
. "$(dirname "$0")/_common.sh"

for k in 2 4 8; do
  echo "### k=$k steps=80000"
  $PY -m runner.stage1_gmm --steps 80000 --k $k --decoder linear \
      --family unrelated --eval-tasks 4 > artifacts/stage1_k${k}_s80k.log 2>&1
done
$PY scripts/ft_reference.py \
  stage1_linear_k2_s80000_unrelated.pt stage1_linear_k4_s80000_unrelated.pt \
  stage1_linear_k8_s80000_unrelated.pt stage1_linear_k16_s80000_unrelated.pt \
  stage1_linear_k32_s80000_unrelated.pt stage1_linear_k64_s80000_unrelated.pt \
  > artifacts/ksweep_80k_full.txt 2>&1
echo "### done"
