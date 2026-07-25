#!/bin/bash
#SBATCH --job-name=pvfm_emu
#SBATCH --partition=general
#SBATCH --output=logs/%j_emu.log
#SBATCH --error=logs/%j_emu.err
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
CKPT=$RUNS/pvfm_lfinit_fixednorm/ckpt_latest.pt
PERSET=$RUNS/mse_e2_perset.npz
EMU=$RUNS/mse_e2_emulator.npz
MEANCURVE=$RUNS/mse_e2_meancurve.npz

# Cosmology-conditioned deployable correction (non-oracle, r-preserving):
#  1) dump per-set (cosmology, T(k)) on 40 extent-2 TRAIN sims
#  2) fit T(k)=f(5-param cosmology) per k-bin (leave-one-out reported)
#  3) apply emulator to extent-2 VAL sims; compare to oracle + train-mean curve
echo "### STEP 1: dump per-set (style,T) on 40 extent-2 TRAIN sims ###"
python3 -u eval_physical.py --ckpt $CKPT --use_ema --steps 1 --split train \
    --sids 16,17,20,21,23,24,25,26,27,30,32,34,35,36,37,41,42,43,44,45,46,47,50,51,52,53,54,55,57,60,61,62,63,64,65,66,67,70,71,72 --max_sets 40 --root $ROOT \
    --dump_tk_perset $PERSET --dump_tk $MEANCURVE

echo "### STEP 2: fit cosmology-conditioned emulator ###"
python3 -u scripts/fit_tk_emulator.py $PERSET $EMU

echo "### STEP 3: apply to extent-2 VAL (emulator vs mean-curve vs oracle) ###"
python3 -u eval_physical.py --ckpt $CKPT --use_ema --steps 1 --split val \
    --sids 28,38,48,58,68 --max_sets 5 --root $ROOT \
    --spectral_correct --correct_curve $MEANCURVE --correct_emulator $EMU
