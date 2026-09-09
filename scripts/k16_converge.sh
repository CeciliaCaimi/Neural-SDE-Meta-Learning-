#!/usr/bin/env bash
# Stage 1: train the document's default k = 16 to the same 320k steps as k = 64.
#
# Without a step-matched run the k comparison confounds coordinate capacity with
# training budget, which is the confusion k_matched.sh was written to avoid at 80k.
set -u
. "$(dirname "$0")/_common.sh"

$PY -m runner.stage1_gmm --steps 320000 --k 16 --decoder linear \
    --family unrelated --eval-tasks 4 > artifacts/stage1_k16_s320k.log 2>&1
$PY scripts/ft_reference.py \
  stage1_linear_k16_s80000_unrelated.pt stage1_linear_k16_s160000_unrelated.pt \
  stage1_linear_k16_s320000_unrelated.pt stage1_linear_k64_s320000_unrelated.pt \
  > artifacts/k16_convergence.txt 2>&1
echo "### done"
