#!/usr/bin/env bash
# 减速是真的表示饱和,还是"大 k 训得更不足"的假象?
# 把 k=32 / k=64 训到 160k(2x),看 oracle 还降不降。
# k=32 是对照: 若两者都继续降,说明 80k 普遍不够;
# 若只有 k=64 降,说明减速是训练量伪影。
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
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
