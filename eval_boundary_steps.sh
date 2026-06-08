#!/bin/bash
#SBATCH --job-name=pvfm_bdy_st
#SBATCH --partition=general
#SBATCH --output=logs/%j_boundary.log
#SBATCH --error=logs/%j_boundary.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate

CKPT="${1:?usage: sbatch eval_boundary_steps.sh <ckpt> <steps> [max_batches]}"
STEPS="${2:?usage: ... <steps>}"
MAX_BATCHES="${3:-16}"

OUT_DIR="$(dirname $CKPT)/boundary_steps${STEPS}_$(basename $CKPT .pt)"

echo "Job start: $(date) on $(hostname)" >&2
echo "ckpt=$CKPT steps=$STEPS max_batches=$MAX_BATCHES out_dir=$OUT_DIR" >&2

python3 -u eval_boundary.py \
    --ckpt "$CKPT" \
    --steps "$STEPS" \
    --max_batches "$MAX_BATCHES" \
    --out_dir "$OUT_DIR" \
    --use_ema \
    --device cuda
