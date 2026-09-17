#!/usr/bin/env bash
# A2: evaluate both conditioning arms at generation, on A1's protocol.
#
# A1's protocol exactly -- 20 val classes x 3 transformations x K_T in {1, 5, 20}, 60
# episodes per K_T, 96 samples per cell, DDIM eta=0 with 50 steps, seed 4321, M_S=64 -- so
# that the cifar_ds3 column A1 and A3 were measured on can sit beside the two new arms.
# Both arms get the identical command, which is what makes the episodes, the supports and
# the initial noise shared and the differences paired.
#
#     bash scripts/a2_eval.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
SPLIT=artifacts/cifar100_domainshift_c3.json
OUT=handoff/results
export PYTHONIOENCODING=utf-8

PROTOCOL="--domainshift-path $SPLIT --split val --n-classes 20 --k-shots 1 5 20
          --n-samples 96 --ddim-steps 50 --seed 4321 --semantic --progress 20"

mkdir -p "$OUT"

for arm in basis film; do
  ck=checkpoints/a2_${arm}_step50000.pt
  if [ ! -f "$ck" ]; then
    echo "missing $ck -- train first (scripts/a2_train.sh)"
    exit 1
  fi
  echo "=== A2 $arm at generation  $(date) ==="
  $PY scripts/gen_strategies.py "$ck" $PROTOCOL \
      --json-out "$OUT/A2_${arm}_per_episode.json" \
      --grid-out "$OUT/A2_${arm}_grid.png" \
      > "$OUT/A2_${arm}_gen.txt" 2> "$OUT/A2_${arm}_gen.progress"
  echo "$arm exit $?"
done

echo "=== the cross-arm table  $(date) ==="
$PY scripts/a2_compare.py --metric sw \
    --also "cifar_ds3 (A1)=handoff/results/A1_per_episode.json" \
    --out "$OUT/A2_basis_vs_film.txt" > /dev/null
$PY scripts/a2_compare.py --metric sig \
    --out "$OUT/A2_basis_vs_film_sig.txt" > /dev/null

echo "=== done  $(date) ==="
ls -la "$OUT"/A2_*
