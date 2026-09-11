#!/usr/bin/env bash
# The driver for Yu Cao's package: the stage-1 positive control and the task-distance sweep.
#
#     bash handoff/Yu-Cao/run_all.sh              # base run + panel, about 30 minutes
#     bash handoff/Yu-Cao/run_all.sh sweep        # the five-point relatedness sweep
#     bash handoff/Yu-Cao/run_all.sh all          # both
#
# CPU only. Nothing here needs a dataset.
set -u
. "$(dirname "$0")/../../scripts/_common.sh"

WHICH="${1:-base}"
STEPS="${STAGE1_STEPS:-80000}"
OUT=handoff/Yu-Cao/results
mkdir -p "$OUT"

need() {
    if [ ! -f "$1" ]; then
        echo "not yet written: $1"
        echo "  see handoff/Yu-Cao/README.md for what it has to do"
        exit 1
    fi
}

# Check before training, not after. The sweep below is two hours; discovering a missing
# analysis script at the end of it is the expensive way to find out.
need scripts/stage1_panel.py

if [ "$WHICH" = "base" ] || [ "$WHICH" = "all" ]; then
    echo "===== stage 1, unrelated family, $STEPS steps ====="
    $PY -m runner.stage1_gmm --steps "$STEPS" --k 16 --decoder linear \
        --family unrelated --eval-tasks 4 2>&1 | tee "artifacts/stage1_panel_base.log"


    echo
    echo "===== four-way panel + coordinate-loss profile ====="
    $PY scripts/stage1_panel.py "checkpoints/stage1_linear_k16_s${STEPS}_unrelated.pt" \
        --profile-out "$OUT/stage1_profile.png" 2>&1 | tee "$OUT/stage1_panel.txt"
fi

if [ "$WHICH" = "sweep" ] || [ "$WHICH" = "all" ]; then
    echo
    echo "===== relatedness sweep -- five settings at $STEPS steps, leave this running ====="
    STAGE1_STEPS="$STEPS" bash scripts/run_relatedness_sweep.sh


    echo
    echo "===== the panel on each of the five checkpoints ====="
    : > "$OUT/stage1_relatedness.txt"
    for ck in stage1_linear_k16_s${STEPS}_related0.1 \
              stage1_linear_k16_s${STEPS}_related0.25 \
              stage1_linear_k16_s${STEPS}_related0.5 \
              stage1_linear_k16_s${STEPS}_related1.0 \
              stage1_linear_k16_s${STEPS}_unrelated; do
        if [ ! -f "checkpoints/${ck}.pt" ]; then
            echo "missing checkpoints/${ck}.pt -- did the sweep finish?"
            continue
        fi
        echo "----- $ck -----" | tee -a "$OUT/stage1_relatedness.txt"
        $PY scripts/stage1_panel.py "checkpoints/${ck}.pt" \
            2>&1 | tee -a "$OUT/stage1_relatedness.txt"
    done
    echo
    echo "Now plot delta_task against perturb, with intervals, into"
    echo "  $OUT/delta_task_vs_distance.png"
fi

echo
echo "===== done ====="
echo "results in $OUT -- these are tracked by git."
echo "artifacts/*.log is tracked too; artifacts/*.txt and *.png are NOT."
ls -la "$OUT" 2>/dev/null || true
