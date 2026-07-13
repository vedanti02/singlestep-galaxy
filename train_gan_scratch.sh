#!/bin/bash
#SBATCH --job-name=pvfm_gan
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-t5-24,babel-m5-28

# Phase-2: from-scratch conditional WGAN-GP (config/gan_scratch.yaml).
# GAN steps are ~2x an MSE step (extra t=0 forward, 3 CIC deposits,
# 2 critics + GP), so this asks for 48h; the script auto-resumes from
# ckpt_latest.pt, so a requeue continues where it stopped.

set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate
echo "Job start: $(date) on $(hostname)" >&2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi >&2

RESUME_ARG=""
CKPT=/data/user_data/vkshirsa/singlestep_runs/pvfm_gan_scratch/ckpt_latest.pt
if [[ -f "$CKPT" ]]; then
    RESUME_ARG="--resume $CKPT"
    echo "Resuming from $CKPT" >&2
fi

python3 -u train.py \
    --config config/gan_scratch.yaml \
    $RESUME_ARG
