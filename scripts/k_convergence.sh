#!/usr/bin/env bash
# Stage 1: is the slowdown real saturation, or an artefact of larger k being less
# trained at a fixed budget?
#
# Train k = 32 and k = 64 to 160k steps (2x) and watch whether the oracle keeps
# falling. k = 32 is the control: if both keep falling, 80k was too short for every
# k; if only k = 64 falls, the slowdown was a training-budget artefact.
set -u
. "$(dirname "$0")/_common.sh"

for k in 32 64; do
  echo "### k=$k steps=160000"
  $PY -m runner.stage1_gmm --steps 160000 --k $k --decoder linear \
      --family unrelated --eval-tasks 4 > artifacts/stage1_k${k}_s160k.log 2>&1
done
$PY scripts/ft_reference.py \
  stage1_linear_k32_s80000_unrelated.pt  stage1_linear_k32_s160000_unrelated.pt \
  stage1_linear_k64_s80000_unrelated.pt  stage1_linear_k64_s160000_unrelated.pt \
  > artifacts/k_convergence.txt 2>&1
echo "### done"
