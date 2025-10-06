#!/bin/bash

set -euo pipefail

########################
# 0) ENV (lightweight) #
########################
# (Keep modules minimal to avoid conflicts. Comment out if your cluster auto-loads.)
module load cuda/12.6.1 || true
module load gcc/11.4.0  || true

# --- Make 'conda activate' work in non-interactive batch shells ---
if ! command -v conda >/dev/null 2>&1; then
  # Try common install locations
  for PFX in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
    if [ -f "$PFX/etc/profile.d/conda.sh" ]; then
      # shellcheck source=/dev/null
      source "$PFX/etc/profile.d/conda.sh"
      break
    fi
  done
fi
# Fallback hook (in case conda.sh exists but 'conda' not in PATH)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" 2>/dev/null || true
else
  echo "[FATAL] conda not found. Please install or module load conda." >&2
  exit 1
fi

conda activate nanogpt || { echo "[FATAL] conda env 'nanogpt' not found."; exit 1; }

cd ~/workspace/nanoGPT

mkdir -p logs runs

#################################
# 1) LAUNCH TRAINING
#################################
python train.py config/train_shakespeare_char.py \
  --csv_path "runs/losses.csv" | tee -a logs/train.log
