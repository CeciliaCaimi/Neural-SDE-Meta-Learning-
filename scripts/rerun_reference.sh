#!/usr/bin/env bash
# Stage 1: recompute the upper-bound reference for two sweeps already run.
#
# The upper bound is full fine-tuning, and fine-tuning overfits a finite target
# sample: run it too long and the score error rises again. The bound is therefore
# taken as the minimum over a budget grid rather than at one budget, and this
# rescores existing checkpoints under that rule. No training is repeated.
set -u
. "$(dirname "$0")/_common.sh"

echo "### relatedness sweep (upper bound now the minimum over the budget grid)"
$PY scripts/ft_reference.py \
  stage1_linear_k16_s80000_related0.1.pt stage1_linear_k16_s80000_related0.25.pt \
  stage1_linear_k16_s80000_related0.5.pt stage1_linear_k16_s80000_related1.0.pt \
  stage1_linear_k16_s80000_unrelated.pt > artifacts/relatedness_reference_v2.txt 2>&1
echo "### k sweep"
$PY scripts/ft_reference.py \
  stage1_linear_k2_s40000.pt stage1_linear_k4_s40000.pt stage1_linear_k8_s40000.pt \
  stage1_linear_k16_s40000.pt stage1_linear_k32_s40000.pt \
  stage1_linear_k64_s40000_unrelated.pt > artifacts/ksweep_reference_v2.txt 2>&1
echo "### done"
