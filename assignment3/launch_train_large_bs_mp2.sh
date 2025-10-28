#!/usr/bin/env bash
# srun --account=bcrc-delta-gpu --qos=bcrc-delta-gpu -p gpuA100x4 -t 1:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gres=gpu:a100:2 --mem=64g bash assignment3/launch_train_large_bs_mp2.sh

set -euo pipefail

########################
# 0) ENV (lightweight) #
########################
module load cuda/12.6.1 >/dev/null 2>&1 || true
module load gcc/11.4.0  >/dev/null 2>&1 || true

# Make `conda activate` work in non-interactive shells (best effort)
if ! command -v conda >/dev/null 2>&1; then
  for PFX in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
    [ -f "$PFX/etc/profile.d/conda.sh" ] && . "$PFX/etc/profile.d/conda.sh" && break
  done
fi
command -v conda >/dev/null 2>&1 && eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate nanogpt || { echo "[FATAL] conda env 'nanogpt' not found."; exit 1; }

cd ~/workspace/nanoGPT
mkdir -p logs runs

#################################
# 1) LAUNCH TRAINING
#################################
python train.py config/train_shakespeare_char_large_bs_mp2.py \
  --csv_path=runs/losses_large_bs_mp2.csv  | tee -a logs/train_large_bs_mp2.log
