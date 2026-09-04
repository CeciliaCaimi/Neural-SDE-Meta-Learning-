#!/usr/bin/env bash
# 相关度扫描: 检验文档 §1.2 那句 "within a related family of tasks" 的分量。
# perturb 0 -> 任务全同; 1 -> 退化回互不相关。
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8

for p in 0.1 0.25 0.5 1.0; do
  echo "### related@$p"
  $PY -m runner.stage1_gmm --steps 80000 --k 16 --decoder linear \
      --family related --perturb $p --eval-tasks 4 \
      > artifacts/stage1_related_p${p}.log 2>&1
done
echo "### unrelated 参照（重跑以统一 checkpoint 元数据）"
$PY -m runner.stage1_gmm --steps 80000 --k 16 --decoder linear \
    --family unrelated --eval-tasks 4 \
    > artifacts/stage1_related_unrelated.log 2>&1

echo "### 上界参照"
$PY scripts/ft_reference.py \
    stage1_linear_k16_s80000_related0.1.pt \
    stage1_linear_k16_s80000_related0.25.pt \
    stage1_linear_k16_s80000_related0.5.pt \
    stage1_linear_k16_s80000_related1.0.pt \
    stage1_linear_k16_s80000_unrelated.pt \
    > artifacts/relatedness_reference.txt 2>&1
echo "### done"
