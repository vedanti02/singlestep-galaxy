#!/bin/bash
#SBATCH --job-name=pvfm_ft
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=babel-t5-24,babel-m5-28

set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate
echo "Job start: $(date) on $(hostname)" >&2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi >&2

# Copy ep9 ckpt into the new run dir to preserve provenance
OUT=runs/pvfm_finetune_$SLURM_JOB_ID
mkdir -p "$OUT"
cp runs/pvfm_noskip_8137633/ckpt_epoch009.pt "$OUT/"

python3 -u train.py \
    --config config/lfinit_noskip_finetune.yaml \
    --override train.out_dir=$OUT \
    --resume runs/pvfm_noskip_8137633/ckpt_epoch009.pt
