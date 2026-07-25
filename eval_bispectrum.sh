#!/bin/bash
#SBATCH --job-name=pvfm_bispec
#SBATCH --partition=general
#SBATCH --output=logs/%j_bispec.log
#SBATCH --error=logs/%j_bispec.err
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
D=$RUNS/bispec_density
SIDS=28,38,48,58,68,78,88,108,128,138,148,158,178,198,208,218   # 16 extent-2 val boxes

# Held-out bispectrum. Reconstruct densities for each model, then compute the
# bispectrum (a statistic never used in training or selection).
echo "### dump densities: MSE / GAN / noz on 16 extent-2 val boxes ###"
python3 -u eval_physical.py --ckpt $RUNS/pvfm_lfinit_fixednorm/ckpt_latest.pt \
    --use_ema --steps 1 --split val --sids $SIDS --max_sets 16 --root $ROOT \
    --dump_density $D/mse
python3 -u eval_physical.py --ckpt $RUNS/pvfm_gan_scratch/ckpt_latest.pt \
    --use_ema --steps 1 --split val --sids $SIDS --max_sets 16 --root $ROOT \
    --dump_density $D/gan
python3 -u eval_physical.py --ckpt $RUNS/pvfm_gan_noz/ckpt_latest.pt \
    --use_ema --steps 1 --split val --sids $SIDS --max_sets 16 --root $ROOT \
    --dump_density $D/noz

echo "### compute + plot bispectrum ###"
python3 -u scripts/compute_bispectrum.py \
    --model MSE:$D/mse --model GAN:$D/gan --model noz:$D/noz \
    --out figures_singlestep/bispectrum.png
