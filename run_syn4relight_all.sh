#!/bin/bash
# One Synthetic4Relight scene per GPU. Edit the list below, then: bash run_syn4relight_all.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# scene:gpu
JOBS=(
    air_baloons:0
    chair:1
    hotdog:2
    jugs:3
)

pids=()
for job in "${JOBS[@]}"; do
    scene="${job%%:*}"
    gpu="${job##*:}"
    echo "[${scene}] starting (GPU ${gpu}) -> logs/syn4_${scene}.log"
    SCENE="${scene}" GPU="${gpu}" bash run_syn4relight.sh > "logs/syn4_${scene}.log" 2>&1 &
    pids+=($!)
done

echo "All scenes launched in background. Waiting..."

status=0
idx=0
for job in "${JOBS[@]}"; do
    scene="${job%%:*}"
    wait "${pids[$idx]}" || { echo "[${scene}] failed (see logs/syn4_${scene}.log)"; status=1; }
    idx=$((idx + 1))
done

if [ "$status" -eq 0 ]; then
    echo "All scenes finished."
else
    echo "Some scenes failed. Check logs/."
fi
exit "$status"
