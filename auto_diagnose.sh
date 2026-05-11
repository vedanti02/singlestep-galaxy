#!/bin/bash
#SBATCH --job-name=pvfm_autodiag
#SBATCH --partition=general
#SBATCH --output=logs/%j_autodiag.log
#SBATCH --error=logs/%j_autodiag.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Watches runs/*/ for new ckpt_epoch*.pt files and runs the boundary
# diagnostic on each (boundary CSV stored next to the checkpoint).
# Stops after `MAX_LOOPS` iterations or `STOP_FILE` exists.

set -euo pipefail
mkdir -p logs

source ~/venv/bin/activate

cd /home/vkshirsa/singlestep-galaxy

DONE_DIR="runs/.diagnosed"
mkdir -p "$DONE_DIR"
STOP_FILE="$DONE_DIR/STOP"
MAX_LOOPS="${MAX_LOOPS:-720}"  # 720 * 60s = 12h

echo "[autodiag] watching runs/*/ckpt_epoch*.pt — touch $STOP_FILE to halt"
loop=0
while [[ $loop -lt $MAX_LOOPS ]]; do
    [[ -f "$STOP_FILE" ]] && { echo "[autodiag] stop file found"; break; }
    new_ckpts=$(find runs -name "ckpt_epoch*.pt" -newer "$DONE_DIR" 2>/dev/null | sort -u)
    for ckpt in $new_ckpts; do
        marker="$DONE_DIR/$(echo "$ckpt" | tr '/' '_')"
        [[ -f "$marker" ]] && continue
        run_dir="$(dirname "$ckpt")"
        epoch="$(basename "$ckpt" .pt | sed 's/ckpt_//')"
        out_dir="$run_dir/boundary/$epoch"
        echo "=== $(date '+%H:%M:%S') diagnosing $ckpt -> $out_dir ==="
        python3 -u eval_boundary.py \
            --ckpt "$ckpt" \
            --max_batches 16 \
            --use_ema \
            --device cuda \
            --out_dir "$out_dir" 2>&1 | tail -8 || echo "[autodiag] failed on $ckpt"
        touch "$marker"
    done
    sleep 60
    loop=$((loop + 1))
done
echo "[autodiag] done after $loop loops"
