#!/bin/bash
#SBATCH --job-name=pvfm_l40s
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
mkdir -p logs

source ~/venv/bin/activate

echo "Job start: $(date) on $(hostname)" >&2
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')" >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

nvidia-smi >&2

# Train the StepOne-PVD model.
# Defaults come from config/default.yaml; pass --override key=value to tweak.
python3 -u train.py \
    --override \
        train.epochs=25 \
        train.batch_size=4 \
        train.num_workers=8 \
        train.device=cuda \
        train.ckpt_every=1 \
        train.out_dir=runs/pvfm_l40s_$SLURM_JOB_ID \
        data.crop_size=32 \
        data.crop_overlap=8 \
        data.env_outside_mask=true \
        model.base_voxel=32 \
        model.base_point=128 \
        model.n_blocks=6 \
        optim.lr=2e-4
