#!/bin/bash
#SBATCH --job-name=pvfm_recon
#SBATCH --partition=general
#SBATCH --output=logs/%j_recon.log
#SBATCH --error=logs/%j_recon.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-t5-24,babel-m5-28,babel-m5-20,babel-s5-28,babel-q5-24
set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate
export PYTHONUNBUFFERED=1
RUNS=/data/user_data/vkshirsa/singlestep_runs
ROOT=/data/user_data/vkshirsa/lagrangian_output_64_fixed
CHUNK=$(cat "$1"); SPLIT="$2"; DEST="$3"
# reconstruct densities (rho_hf, rho_lf, rho_pred=MSE-SR, style) for the KL posterior test.
# Retry on the transient "CUDA device busy/unavailable" startup error.
for attempt in 1 2 3 4; do
  if python3 -u eval_physical.py --ckpt $RUNS/pvfm_lfinit_fixednorm/ckpt_latest.pt \
       --use_ema --steps 1 --split "$SPLIT" --sids "$CHUNK" --max_sets 200 \
       --root $ROOT --dump_density "$DEST"; then
    exit 0
  fi
  echo "[recon] attempt $attempt failed; sleeping 60s and retrying" >&2
  sleep 60
done
exit 1
