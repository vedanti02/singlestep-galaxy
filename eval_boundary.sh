#!/bin/bash
#SBATCH --job-name=pvfm_boundary
#SBATCH --partition=general
#SBATCH --output=logs/%j_boundary.log
#SBATCH --error=logs/%j_boundary.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Boundary-error diagnostic for a trained PVFM checkpoint.
# Usage:
#   sbatch eval_boundary.sh <CKPT> [MAX_BATCHES]
# Example (smoke test, 8 batches):
#   sbatch eval_boundary.sh runs/pvfm_a100_7556358/ckpt_latest.pt 8
# Example (full validation pass):
#   sbatch eval_boundary.sh runs/pvfm_a100_7556358/ckpt_latest.pt

set -euo pipefail
mkdir -p logs

source ~/venv/bin/activate

CKPT="${1:?usage: sbatch eval_boundary.sh <ckpt> [max_batches]}"
MAX_BATCHES="${2:-}"

echo "Job start: $(date) on $(hostname)" >&2
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')" >&2
echo "ckpt=$CKPT  max_batches=${MAX_BATCHES:-<all>}" >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

nvidia-smi >&2

if [[ -n "$MAX_BATCHES" ]]; then
    python3 -u eval_boundary.py \
        --ckpt "$CKPT" \
        --max_batches "$MAX_BATCHES" \
        --use_ema \
        --device cuda
else
    python3 -u eval_boundary.py \
        --ckpt "$CKPT" \
        --use_ema \
        --device cuda
fi

echo "Job end: $(date)" >&2
