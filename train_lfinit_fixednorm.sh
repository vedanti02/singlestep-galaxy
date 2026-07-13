#!/bin/bash
#SBATCH --job-name=pvfm_fixednorm
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=babel-t5-24,babel-m5-28

# Phase-1 re-baseline: lfinit_noskip on the FIXED preprocessing
# (tile-based norm stats, residual-std target, regenerated stitched/).
# See research_scratch_pad.md.

set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate
echo "Job start: $(date) on $(hostname)" >&2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi >&2

RESUME_ARG=""
CKPT=/data/user_data/vkshirsa/singlestep_runs/pvfm_lfinit_fixednorm/ckpt_latest.pt
if [[ -f "$CKPT" ]]; then
    RESUME_ARG="--resume $CKPT"
    echo "Resuming from $CKPT" >&2
fi

python3 -u train.py \
    --config config/lfinit_noskip_fixednorm.yaml \
    $RESUME_ARG
