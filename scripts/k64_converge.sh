#!/usr/bin/env bash
# 捕获比例到底收敛在哪? 320k 步。
# 若继续爬向 100%,说明低维形式没问题,之前的"不通过"纯属训练不足;
# 若在某个明显低于 100% 的值上平掉,那才是对该形式的定量限制。
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY -m runner.stage1_gmm --steps 320000 --k 64 --decoder linear \
    --family unrelated --eval-tasks 4 > artifacts/stage1_k64_s320k.log 2>&1
$PY scripts/ft_reference.py \
  stage1_linear_k64_s80000_unrelated.pt stage1_linear_k64_s160000_unrelated.pt \
  stage1_linear_k64_s320000_unrelated.pt > artifacts/k64_convergence.txt 2>&1
echo "### done"
