#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"

mkdir -p logs

echo "[armadillo] starting (GPU 4) -> logs/tensoir_armadillo.log"
bash run_tensoir_armadillo.sh > logs/tensoir_armadillo.log 2>&1 &
pid_armadillo=$!

echo "[ficus] starting (GPU 5) -> logs/tensoir_ficus.log"
bash run_tensoir_ficus.sh > logs/tensoir_ficus.log 2>&1 &
pid_ficus=$!

echo "[hotdog] starting (GPU 6) -> logs/tensoir_hotdog.log"
bash run_tensoir_hotdog.sh > logs/tensoir_hotdog.log 2>&1 &
pid_hotdog=$!

echo "[lego] starting (GPU 7) -> logs/tensoir_lego.log"
bash run_tensoir_lego.sh > logs/tensoir_lego.log 2>&1 &
pid_lego=$!

echo "All scenes launched in background. Waiting..."

status=0
wait "$pid_armadillo" || { echo "[armadillo] failed (see logs/tensoir_armadillo.log)"; status=1; }
wait "$pid_ficus" || { echo "[ficus] failed (see logs/tensoir_ficus.log)"; status=1; }
wait "$pid_hotdog" || { echo "[hotdog] failed (see logs/tensoir_hotdog.log)"; status=1; }
wait "$pid_lego" || { echo "[lego] failed (see logs/tensoir_lego.log)"; status=1; }

if [ "$status" -eq 0 ]; then
    echo "All scenes finished."
else
    echo "Some scenes failed. Check logs/."
fi
exit "$status"
