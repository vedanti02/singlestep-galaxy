#!/bin/bash
#SBATCH --job-name=pvfm_dispcritic
#SBATCH --partition=general
#SBATCH --output=logs/%j_train.log
#SBATCH --error=logs/%j_error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=babel-t5-24,babel-m5-28,babel-m5-20,babel-s5-28,babel-q5-24

# GAN ablation: dispcritic. See research_scratch_pad.md.
# ROOT-CAUSE FIX for dataloader I/O starvation: tiles are ~38k tiny 3MB .npy
# on NFS with 0.57s/tile random-read latency that starves the GPU. Page-cache
# warming degraded (cache reclaimed under shared-node mem pressure: SLURM sets
# a cgroup mem LIMIT, not a memory.min floor). Node-local /scratch NVMe staging
# is durable and non-reclaimable -> fast reads after a one-time copy.
set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate
echo "Job start: $(date) on $(hostname)" >&2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
nvidia-smi >&2

SRCROOT=/data/user_data/vkshirsa/lagrangian_output_64_fixed
SDIR=/scratch/$USER/pvfm_stage
mkdir -p /scratch/$USER
exec 9>"/scratch/$USER/.pvfm_stage.lock"
flock 9
if [ ! -f "$SDIR/.complete" ]; then
  echo "[stage] staging ~180G NFS->local $SDIR ..." >&2
  t0=$SECONDS
  mkdir -p "$SDIR"
  rsync -aL "$SRCROOT/quijotelike-64/" "$SDIR/quijotelike-64/"
  rsync -aL --exclude='vel.npy' "$SRCROOT/quijote-64/" "$SDIR/quijote-64/"
  rsync -aL "$SRCROOT/stitched/" "$SDIR/stitched/"
  touch "$SDIR/.complete"
  echo "[stage] done in $((SECONDS-t0))s" >&2
else
  echo "[stage] reusing existing $SDIR" >&2
fi
flock -u 9
df -h /scratch >&2

CKPT=/data/user_data/vkshirsa/singlestep_runs/pvfm_gan_dispcritic/ckpt_latest.pt
RESUME_ARG=""
[[ -f "$CKPT" ]] && RESUME_ARG="--resume $CKPT" && echo "Resuming from $CKPT" >&2

python3 -u train.py --config config/gan_dispcritic.yaml \
    --override data.root=$SDIR \
    $RESUME_ARG
