#!/usr/bin/env bash
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
echo "### 相关度扫描（上界改为预算网格最小值）"
$PY scripts/ft_reference.py \
  stage1_linear_k16_s80000_related0.1.pt stage1_linear_k16_s80000_related0.25.pt \
  stage1_linear_k16_s80000_related0.5.pt stage1_linear_k16_s80000_related1.0.pt \
  stage1_linear_k16_s80000_unrelated.pt > artifacts/relatedness_reference_v2.txt 2>&1
echo "### k 扫描"
$PY scripts/ft_reference.py \
  stage1_linear_k2_s40000.pt stage1_linear_k4_s40000.pt stage1_linear_k8_s40000.pt \
  stage1_linear_k16_s40000.pt stage1_linear_k32_s40000.pt \
  stage1_linear_k64_s40000_unrelated.pt > artifacts/ksweep_reference_v2.txt 2>&1
echo "### done"
