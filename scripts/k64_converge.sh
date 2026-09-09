#!/usr/bin/env bash
# Stage 1: where does the captured fraction actually converge? 320k steps.
#
# If it keeps climbing toward 100%, the low-dimensional form is sound and the earlier
# failing verdict was simply undertraining. If it levels off somewhere clearly below
# 100%, that plateau is a quantitative limit on the form itself.
set -u
. "$(dirname "$0")/_common.sh"

$PY -m runner.stage1_gmm --steps 320000 --k 64 --decoder linear \
    --family unrelated --eval-tasks 4 > artifacts/stage1_k64_s320k.log 2>&1
$PY scripts/ft_reference.py \
  stage1_linear_k64_s80000_unrelated.pt stage1_linear_k64_s160000_unrelated.pt \
  stage1_linear_k64_s320000_unrelated.pt > artifacts/k64_convergence.txt 2>&1
echo "### done"
