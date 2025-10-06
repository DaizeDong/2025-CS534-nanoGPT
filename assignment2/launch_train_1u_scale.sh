#!/usr/bin/env bash
# assignment2/launch_train_1u_scale.sh
# Fix GPU grouping (parse ONLY small integer indices), support decimal scale,
# round BS to int (>=1), LR to 3 decimals, and pack multiple DDP jobs.
# Keep it simple and runnable.

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
# 1) BASE HYPER-PARAMS (simple) #
#################################
BASE_BATCH_SIZE=64          # per-process micro-batch; will be scaled then rounded
BASE_LEARNING_RATE="1e-3"   # sci-notation ok
CONFIG_FILE="config/train_shakespeare_char.py"

#############################################
# 2) USER KNOBS (via env or defaults below) #
#############################################
# Examples:
#   GPUS=0,1,2,3 NPROC=2 SCALES="0.5 1 2 4" ./assignment2/launch_train_1u_scale.sh
GPUS="${GPUS:-0,1,2,3}"              # comma/space-separated cuda device indices
NPROC="${NPROC:-1}"                  # DDP processes per job (group size)
SCALES="${SCALES:-7.0 8.0 12.0 16.0}" # 0.25 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0

#################################
# 3) Small helper math/format   #
#################################
_round_to_int () { awk -v v="$1" 'BEGIN{n=int(v+0.5); if(n<1)n=1; printf "%d", n}'; }
_fmt3        () { awk -v v="$1" 'BEGIN{printf "%.3f", v+0.0}'; }
_mul         () { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.12f", a*b}'; }

# Strictly parse GPU list: keep ONLY non-negative integers < 128; drop everything else.
# Accept commas/spaces as separators; de-duplicate and sort ascending.
_parse_gpu_list () {
  local raw="$1"
  # normalize separators -> newlines
  printf '%s\n' "$raw" | tr ' ,' '\n' \
  | awk '/^[0-9]+$/ && $1 < 128 {print $1}' \
  | sort -n | uniq \
  | paste -sd',' -
}

# Auto-detect GPUs via nvidia-smi (indices only)
_autodetect_gpus () {
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  nvidia-smi --query-gpu=index --format=csv,noheader \
    | tr -d ' ' \
    | sort -n | uniq \
    | paste -sd',' -
}

# Split CUDA device list into groups of size NPROC
_split_into_groups () {
  local list_csv="$1" group_sz="$2"
  local IFS=','
  local -a ALL=()
  read -r -a ALL <<< "$list_csv"

  local total="${#ALL[@]}"
  local i=0
  GPU_GROUPS=()  # NOTE: do NOT name this 'GROUPS' (Bash builtin)
  while (( i < total )); do
    local -a g=()
    for ((k=0; k<group_sz && i<total; k++, i++)); do g+=("${ALL[$i]}"); done
    (( ${#g[@]} == group_sz )) && GPU_GROUPS+=("$(IFS=','; echo "${g[*]}")")
  done
  (( ${#GPU_GROUPS[@]} > 0 )) || { echo "[FATAL] Need at least ${group_sz} GPUs. Provided: ${list_csv}" >&2; exit 1; }
}

#################################
# 4) One-job launcher (DDP)     #
#################################
launch_one () {
  local dev_csv="$1" scale="$2"
  local bs_float lr_float bs_int lr_fmt log_sfx mport

  bs_float="$(_mul "$BASE_BATCH_SIZE"    "$scale")"
  lr_float="$(_mul "$BASE_LEARNING_RATE" "$scale")"
  bs_int="$(_round_to_int "$bs_float")"
  lr_fmt="$(_fmt3 "$lr_float")"
  mport=$(( 12340 + RANDOM % 2000 ))   # avoid port collisions

  log_sfx="${NPROC}u_s${scale}_bs${bs_int}_lr${lr_fmt}"
  printf '[LAUNCH] DEVICES={%s} SCALE=%s BS=%d LR=%s PORT=%d\n' "$dev_csv" "$scale" "$bs_int" "$lr_fmt" "$mport"

  CUDA_VISIBLE_DEVICES="$dev_csv" MASTER_PORT="$mport" \
  torchrun --standalone --nproc_per_node="$NPROC" train.py "$CONFIG_FILE" \
    --batch_size="$bs_int" \
    --learning_rate="$lr_fmt" \
    --csv_path="runs/${log_sfx}.csv" \
    --out_dir="out/${log_sfx}" \
    2>&1 | tee -a "logs/train_${log_sfx}.log" &
}

########################
# 5) Pack & dispatch   #
########################
# If GPUS looks invalid (e.g., user accidentally set it to `nvidia-smi` output),
# strictly parse it; if result empty, auto-detect.
PARSED_GPUS="$(_parse_gpu_list "$GPUS")"
if [ -z "$PARSED_GPUS" ]; then
  PARSED_GPUS="$(_autodetect_gpus)"
fi
[ -n "$PARSED_GPUS" ] || { echo "[FATAL] Could not determine GPU indices. GPUS='$GPUS'"; exit 1; }

# Basic sanity: NPROC must be positive integer
[[ "$NPROC" =~ ^[1-9][0-9]*$ ]] || { echo "[FATAL] NPROC must be a positive integer. Got: ${NPROC}"; exit 1; }

_split_into_groups "$PARSED_GPUS" "$NPROC"
GPU_NUM_GROUPS="${#GPU_GROUPS[@]}"

printf '[INFO] GPUS(raw)  = "%s"\n' "$GPUS"
printf '[INFO] GPUS(parsed)="%s"\n' "$PARSED_GPUS"
printf '[INFO] GPU GROUPS (%d x %d GPUs):\n' "$GPU_NUM_GROUPS" "$NPROC"
for g in "${GPU_GROUPS[@]}"; do printf '  - {%s}\n' "$g"; done
printf '[INFO] SCALES: %s\n' "$SCALES"
printf '[INFO] CONFIG: %s\n\n' "$CONFIG_FILE"

# Launch in waves; each wave runs up to GPU_NUM_GROUPS jobs concurrently
read -r -a PENDING_SCALES <<< "$SCALES"
idx=0
while (( idx < ${#PENDING_SCALES[@]} )); do
  echo "[WAVE] launching up to ${GPU_NUM_GROUPS} DDP jobs..."
  for (( gi=0; gi<GPU_NUM_GROUPS && idx<${#PENDING_SCALES[@]}; gi++, idx++ )); do
    launch_one "${GPU_GROUPS[$gi]}" "${PENDING_SCALES[$idx]}"
    sleep 1
  done
  echo "[WAVE] waiting..."
  wait
  echo "[WAVE] done."
done

echo "[DONE] all experiments finished."
