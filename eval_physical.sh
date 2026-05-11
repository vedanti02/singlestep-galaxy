#!/bin/bash
#SBATCH --job-name=pvfm_phys
#SBATCH --partition=general
#SBATCH --output=logs/%j_phys.log
#SBATCH --error=logs/%j_phys.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Cosmological evaluation: transfer function + coherence vs ground truth.
# Usage:
#   sbatch eval_physical.sh <CKPT> [MAX_SETS]

set -euo pipefail
mkdir -p logs

source ~/venv/bin/activate

CKPT="${1:?usage: sbatch eval_physical.sh <ckpt> [max_sets]}"
MAX_SETS="${2:-2}"

echo "Job start: $(date) on $(hostname)" >&2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi >&2

python3 -u eval_physical.py \
    --ckpt "$CKPT" \
    --max_sets "$MAX_SETS" \
    --use_ema \
    --device cuda
