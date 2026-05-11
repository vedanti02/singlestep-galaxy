#!/bin/bash
#SBATCH --job-name=pvfm_infer
#SBATCH --partition=general
#SBATCH --output=logs/%j_infer.log
#SBATCH --error=logs/%j_infer.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Multi-GPU distributed inference for the Single-step PVFM upsampler.
# Usage:
#   sbatch infer.sh <CKPT> <SET_ID> <OUTPUT_H5>
# Example:
#   sbatch infer.sh runs/pvfm_a100_123456/ckpt_latest.pt 9 /scratch/hf_set9.h5

set -euo pipefail
mkdir -p logs

source ~/venv/bin/activate

CKPT="${1:?usage: sbatch infer.sh <ckpt> <set_id> <output_h5>}"
SET_ID="${2:?usage: sbatch infer.sh <ckpt> <set_id> <output_h5>}"
OUT="${3:?usage: sbatch infer.sh <ckpt> <set_id> <output_h5>}"
LF_ROOT="${LF_ROOT:-/data/group_data/universedata/lagrangian_output_64}"
NPROC="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"

echo "Job start: $(date) on $(hostname)" >&2
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')" >&2
echo "ckpt=$CKPT  set_id=$SET_ID  out=$OUT  nproc=$NPROC" >&2

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# NCCL: keep things calm and verbose enough to debug a hang.
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1

nvidia-smi >&2

# Single-node multi-GPU launch. --standalone picks a free port automatically.
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    inference_distributed.py \
        --ckpt_path "$CKPT" \
        --lf_input_path "$LF_ROOT" \
        --set_id "$SET_ID" \
        --output_path "$OUT" \
        --use_ema \
        --temp_dir "${OUT}.tmp"

echo "Job end: $(date)" >&2
