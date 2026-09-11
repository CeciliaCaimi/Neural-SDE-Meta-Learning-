#!/usr/bin/env bash
# The driver for Jing Peng's package: four centred-coordinate runs and the comparison arm.
#
#     bash handoff/Jing-Peng/run_all.sh            # everything, in sequence
#     bash handoff/Jing-Peng/run_all.sh small      # only the 32-channel pair, under an hour
#     bash handoff/Jing-Peng/run_all.sh film       # only the comparison arm
#
# The runs are independent. On a multi-GPU instance, prefer starting them in parallel
# with CUDA_VISIBLE_DEVICES=<i> rather than waiting on this script.
set -u
. "$(dirname "$0")/../../scripts/_common.sh"

WHICH="${1:-all}"
OUT=handoff/Jing-Peng/results
mkdir -p "$OUT"
DS=artifacts/cifar100_domainshift_c3.json

if [ -z "${CIFAR100_ROOT:-}" ]; then
    echo "CIFAR100_ROOT is not set -- run setup.sh, then export the line it prints"
    exit 1
fi

run () {                      # run <name> <extra args...>
    name="$1"; shift
    echo "===== $name ====="
    $PY -m runner.train --scheme domainshift --domainshift-path "$DS" \
        --run-name "$name" "$@" 2>&1 | tee "$OUT/${name}_console.txt"
    cp "checkpoints/${name}_log.jsonl" "$OUT/" 2>/dev/null || \
        echo "warning: no log written for $name"
}

if [ "$WHICH" = "all" ] || [ "$WHICH" = "small" ]; then
    run ctr32_off --k 32 --base-channels 32 --steps 60000
    run ctr32_on  --k 32 --base-channels 32 --steps 60000 --center-coords
    echo
    echo "Look at delta_task in the two logs above before continuing."
    echo "Continue in either case: positive confirms it on the protocol backbone,"
    echo "negative rules out a small-backbone artefact."
fi

if [ "$WHICH" = "all" ]; then
    run ctr128_off --k 16 --base-channels 128 --steps 50000
    run ctr128_on  --k 16 --base-channels 128 --steps 50000 --center-coords
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "film" ]; then
    if [ ! -f runner/train_film.py ]; then
        echo "not yet written: runner/train_film.py -- see handoff/Jing-Peng/README.md"
        exit 1
    fi
    echo "===== film128  (generic latent conditioning, centring off) ====="
    $PY -m runner.train_film --scheme domainshift --domainshift-path "$DS" \
        --run-name film128 --k 16 --steps 50000 2>&1 | tee "$OUT/film128_console.txt"
    cp checkpoints/film128_log.jsonl "$OUT/" 2>/dev/null || true
fi

echo
echo "===== done ====="
echo "results in $OUT. The *_log.jsonl copies are the primary evidence -- checkpoints/"
echo "is git-ignored, so the originals do not travel. Checkpoints stay on this server."
ls -la "$OUT"
