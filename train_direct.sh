#!/bin/bash
#SBATCH --job-name=pvfm_direct
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Direct-regression variant: skips the noise + Euler floor of FM by
# training the model to predict the residual directly. Algorithm
# (regions + d/2 buffer + outside conditioning) unchanged.

set -euo pipefail
mkdir -p logs

source ~/venv/bin/activate

echo "Job start: $(date) on $(hostname)" >&2
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')" >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

nvidia-smi >&2

python3 -u train.py \
    --config config/direct.yaml \
    --override train.out_dir=runs/pvfm_direct_$SLURM_JOB_ID
