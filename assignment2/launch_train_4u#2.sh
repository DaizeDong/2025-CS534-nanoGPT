#!/bin/bash
#SBATCH -A bcrc-dtai-gh
#SBATCH --qos=bcrc-dtai-gh
#SBATCH -p ghx4
#SBATCH -t 12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 每节点 1 个 torchrun 控制进程
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=512g
#SBATCH --job-name=nanoGPT-2x4
#SBATCH --output=%x-%j.out

# 2 节点 × 每节点 4 GPU（总 8 GPU）运行 nanoGPT 的 sbatch 脚本

set -euo pipefail

########################
# 0) ENV (login shell) #
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

cd "$HOME/workspace/nanoGPT"
mkdir -p logs runs

#################################
# 1) DISTRIBUTED CONFIG
#################################
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR MASTER_PORT

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-^lo,docker0}

echo "[INFO] SLURM_NNODES=$SLURM_NNODES"
echo "[INFO] SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] WORLD_SIZE=$((SLURM_NNODES * 4)) (2 nodes × 4 GPUs)"

#################################
# 2) LAUNCH TRAINING
#################################
# 关键修复：在 srun 子进程（每个节点）里再次加载 module/conda，确保能找到 torchrun
now_str=$(date +'%Y%m%d_%H%M%S')

srun -l bash -lc '
  set -euo pipefail

  # --- per-node env rehydrate ---
  module load cuda/12.6.1 >/dev/null 2>&1 || true
  module load gcc/11.4.0  >/dev/null 2>&1 || true

  if ! command -v conda >/dev/null 2>&1; then
    for PFX in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
      [ -f "$PFX/etc/profile.d/conda.sh" ] && . "$PFX/etc/profile.d/conda.sh" && break
    done
  fi
  command -v conda >/dev/null 2>&1 && eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate nanogpt || true
  [ -n "${CONDA_PREFIX:-}" ] && export PATH="$CONDA_PREFIX/bin:$PATH"

  cd "$HOME/workspace/nanoGPT"

  # 训练
  now_str=$(date +"%Y%m%d_%H%M%S")
  torchrun --nproc_per_node=4 \
           --nnodes=$SLURM_NNODES \
           --node_rank=$SLURM_PROCID \
           --master_addr=$MASTER_ADDR \
           --master_port=$MASTER_PORT \
           train.py config/train_shakespeare_char_8u.py \
             --csv_path=runs/losses_8u_${now_str}.csv
' |& tee -a "logs/train_8u_${now_str}.log"
