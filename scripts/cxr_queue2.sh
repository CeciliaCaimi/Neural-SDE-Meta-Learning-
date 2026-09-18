#!/usr/bin/env bash
# Stage C on ChestX-ray14, after the capacity sweep: select, C3, the FiLM arm, the fine-tuning
# budget, then C4. Every step stops the queue on failure -- nothing downstream runs on the
# output of a step that did not finish.
#
# Decided with the user on 2026-09-18, after the Stage C audit:
#   1. FiLM + transport is included (plan C4 arm 6; section 11 puts it in the minimum baseline
#      set for any headline comparison). Trained at the chosen capacity for the same number of
#      steps, and its checkpoint selected by the same validation rule.
#   2. Full fine-tuning is included with its budget chosen on the validation tasks, per K_T.
#      LoRA / adapters do not exist here and are stated as absent, not built.
#   3. The AP/PA confound is reported beside every age reading; PA-only is decided after C3.
#
#     bash scripts/cxr_queue2.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=handoff/results
SPLIT=artifacts/chestxray_split_v2.json
export PYTHONIOENCODING=utf-8
export CHESTXRAY_ROOT="${CHESTXRAY_ROOT:-C:/T1D Meta Learning/dataset/chestxray14}"

step () { echo "=== $1  $(date) ==="; }
die () { echo "FAILED at: $1 -- queue stopped"; exit 1; }
field () { $PY -c "import json;print(json.load(open('$1'))['$2'])"; }

step "waiting for the capacity sweep and the frozen instruments"
while [ ! -f checkpoints/cxr2_64_step30000.pt ] || [ ! -f checkpoints/cxr_instruments.pt ]; do
  sleep 60
done
sleep 30

step "1. model selection on validation"
$PY scripts/cxr_select.py cxr2_128 cxr2_64 --out $OUT/C_cxr_model_selection || die "selection"
CK=$(field $OUT/C_cxr_model_selection.json ckpt)
CH=$(field $OUT/C_cxr_model_selection.json channels)
ST=$(field $OUT/C_cxr_model_selection.json steps_total)
echo "chosen $CK ($CH channels, $ST steps)"

step "2. C3 sanity check"
$PY scripts/c4_cxr.py "$CK" --mode sanity > $OUT/C3_cxr_console.txt 2> $OUT/C3_cxr_progress.txt \
  || die "C3"

step "3. FiLM arm at $CH channels, $ST steps"
$PY -m runner.train_film --scheme chestxray --cxr-path $SPLIT --k 16 --transport-kind linear \
    --m-source 64 --base-channels "$CH" --steps "$ST" --diagnose-every 500 --ckpt-every 1000 \
    --film-mode per_block --run-name cxr2_film > $OUT/C_cxr_train_film.log 2>&1 || die "FiLM train"
$PY scripts/cxr_select.py cxr2_film --out $OUT/C_cxr_film_selection || die "FiLM selection"
FCK=$(field $OUT/C_cxr_film_selection.json ckpt)

step "4. full fine-tuning budget, selected on validation"
$PY scripts/c4_cxr.py "$CK" --mode select-ft > $OUT/C4_cxr_ft_select_console.txt \
    2> $OUT/C4_cxr_ft_select_progress.txt || die "fine-tuning budget"

step "5. C4 on the test tasks"
$PY scripts/c4_cxr.py "$CK" --mode c4 --film-ckpt "$FCK" --ft-budget $OUT/C4_cxr_ft_budget.json \
    > $OUT/C4_cxr_console.txt 2> $OUT/C4_cxr_progress.txt || die "C4"

step "done"
