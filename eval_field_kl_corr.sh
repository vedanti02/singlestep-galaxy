#!/bin/bash
#SBATCH --job-name=pvfm_fklcorr
#SBATCH --partition=general
#SBATCH --output=logs/%j_fklcorr.log
#SBATCH --error=logs/%j_fklcorr.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-t5-24,babel-m5-28,babel-m5-20,babel-s5-28,babel-q5-24
set -euo pipefail
mkdir -p logs
source ~/venv/bin/activate
export PYTHONUNBUFFERED=1
B=/data/user_data/vkshirsa/singlestep_runs
echo "train boxes: $(ls $B/kl_density/train/*.npz 2>/dev/null | wc -l)  test: $(ls $B/kl_density/test/*.npz 2>/dev/null | wc -l)"
python3 -u scripts/field_posterior.py \
    --train_dir $B/kl_density/train_corr --test_dir $B/kl_density/test_corr \
    --res 64 --epochs 60 --seeds 3 \
    --out figures_singlestep/field_kl_corr.png
