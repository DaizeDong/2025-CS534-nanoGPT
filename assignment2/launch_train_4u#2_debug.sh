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
#SBATCH --kill-on-invalid-dep=yes

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

# PyTorch/NCCL 建议变量
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # 新变量，替代 NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_TIMEOUT=1800                # 30 分钟，便于观测（稳定后可去掉/调小）
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="logs/nccl_%h_%p.log"

echo "[INFO] SLURM_NNODES=$SLURM_NNODES"
echo "[INFO] SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] WORLD_SIZE=$((SLURM_NNODES * 4)) (2 nodes x 4 GPUs)"

#################################
# 2) LAUNCH TRAINING (use heredoc to avoid quote issues)
#################################
srun -l --kill-on-bad-exit=1 bash -s <<'SRUN_SCRIPT'
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

# -------- 安全模式（TCP/Socket）--------
# 1) 解析 MASTER_ADDR -> IPv4，确保 ip route get 能工作
MASTER_IP="$(getent ahostsv4 "$MASTER_ADDR" | awk 'NR==1{print $1}')"
if [ -z "${MASTER_IP:-}" ]; then
  MASTER_IP="$(getent hosts "$MASTER_ADDR" | awk 'NR==1{print $1}')"
fi
IFACE="$(ip -o route get "$MASTER_IP" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
if [ -z "${IFACE:-}" ]; then
  # Fallback: default route to public internet
  IFACE="$(ip -o route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
fi
if [ -z "${IFACE:-}" ]; then
  echo "[FATAL] cannot determine NIC iface for MASTER_ADDR=$MASTER_ADDR (MASTER_IP=${MASTER_IP:-<nil>})"
  exit 2
fi
export NCCL_SOCKET_IFNAME="$IFACE"
# 2) 强制 Socket/TCP，禁用 RDMA/IB
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
# 3) 简化协议以规避边角 Bug（稳定后可注释）
export NCCL_PROTO=Simple
export NCCL_CROSS_NIC=0

echo "[NODE ${SLURMD_NODENAME:-unknown}] MASTER_IP=${MASTER_IP:-<nil>} IFACE=$IFACE ; NCCL_NET=${NCCL_NET:-<default>} ; IB_DISABLE=${NCCL_IB_DISABLE:-unset}"

cd "$HOME/workspace/nanoGPT"

# 兼容旧 PyTorch：没有 torchrun 可执行名则回退 python -m
TORCHRUN_BIN="$(python - <<'PY'
import shutil
print(shutil.which("torchrun") or "")
PY
)"
if [ -z "$TORCHRUN_BIN" ]; then
  TORCHRUN_BIN="python -m torch.distributed.run"
fi

# 训练
$TORCHRUN_BIN --nproc_per_node=4 --nnodes="$SLURM_NNODES" \
              --node_rank="$SLURM_PROCID" \
              --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
              train.py config/train_shakespeare_char_8u.py \
                --csv_path=runs/losses_8u_debug.csv
SRUN_SCRIPT
