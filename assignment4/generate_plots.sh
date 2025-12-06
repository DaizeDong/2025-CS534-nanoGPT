#!/bin/bash

# Script to generate all plots for assignment4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Create results directories
mkdir -p "${RESULTS_DIR}/task2"
mkdir -p "${RESULTS_DIR}/task4"

# Task 1 & 2: TP4 vs DP4 Comparison
echo "Generating Task 1 & 2 plots (TP4 vs DP4)..."
python "${SCRIPT_DIR}/plot_pro.py" \
    "${RUNS_DIR}/losses_large_bs_head8_tp4_rank0_no_eval.csv" \
    "${RUNS_DIR}/losses_large_bs_head8_dp4_rank0_no_eval.csv" \
    --labels "TP4" "DP4" \
    --out-step-loss "${RESULTS_DIR}/task2/loss_step.png" \
    --out-time-loss "${RESULTS_DIR}/task2/loss_time.png" \
    --out-time-compare "${RESULTS_DIR}/task2/runtime_comparison.png" \
    --out-mem-avg-bar "${RESULTS_DIR}/task2/memory_comparison.png"

# Task 3 & 4: DP2+TP4 vs DP8 Comparison
echo "Generating Task 3 & 4 plots (DP2+TP4 vs DP8)..."
if [ -f "${RUNS_DIR}/losses_large_bs_head8_tp4_dp2_rank0_no_eval.csv" ]; then
    python "${SCRIPT_DIR}/plot_pro.py" \
        "${RUNS_DIR}/losses_large_bs_head8_tp4_dp2_rank0_no_eval.csv" \
        "${RUNS_DIR}/losses_large_bs_head8_dp8_rank0_no_eval.csv" \
        --labels "DP2+TP4" "DP8" \
        --out-step-loss "${RESULTS_DIR}/task4/loss_step.png" \
        --out-time-loss "${RESULTS_DIR}/task4/loss_time.png" \
        --out-time-compare "${RESULTS_DIR}/task4/runtime_comparison.png" \
        --out-mem-avg-bar "${RESULTS_DIR}/task4/memory_comparison.png"
else
    echo "Warning: ${RUNS_DIR}/losses_large_bs_head8_tp4_dp2_rank0_no_eval.csv not found, skipping Task 3 & 4"
fi

echo "Done generating all plots!"

