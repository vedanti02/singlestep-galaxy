#!/bin/bash
#SBATCH --job-name=pvfm_noskip
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Avoid babel-t5-24 — observed pinned at load avg ~19 by another user's
# multi-GPU TTS training (30+ data workers) starving our dataloader.
#SBATCH --exclude=babel-t5-24,babel-m5-28

set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate
echo "Job start: $(date) on $(hostname)" >&2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi >&2

python3 -u train.py \
    --config config/lfinit_noskip.yaml \
    --override train.out_dir=runs/pvfm_noskip_$SLURM_JOB_ID
