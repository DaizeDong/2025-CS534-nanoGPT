#!/bin/bash
# srun -A bcrc-dtai-gh --qos=bcrc-dtai-gh -p ghx4 -t 12:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:h100:4 --mem=512g --pty bash -i

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
torchrun --standalone --nproc_per_node=4 \
  train.py config/train_shakespeare_char_4u_no_warmup.py \
    --csv_path=runs/losses_4u_no_warmup.csv  | tee -a logs/train_4u_no_warmup.log
