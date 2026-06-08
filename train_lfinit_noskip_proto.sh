#!/bin/bash
#SBATCH --job-name=pvfm_proto
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# Avoid the worst-contended node observed so far.
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
    --config config/lfinit_noskip_proto.yaml \
    --override train.out_dir=runs/pvfm_proto_$SLURM_JOB_ID
