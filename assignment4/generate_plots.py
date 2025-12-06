#!/usr/bin/env python3
"""
Script to generate all plots for assignment4
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RUNS_DIR = SCRIPT_DIR / "runs"
RESULTS_DIR = SCRIPT_DIR / "results"
PLOT_SCRIPT = SCRIPT_DIR / "plot_pro.py"

# Create results directories
(RESULTS_DIR / "task2").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "task4").mkdir(parents=True, exist_ok=True)


def run_plot(args):
    """Run plot_pro.py with given arguments."""
    cmd = [sys.executable, str(PLOT_SCRIPT)] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"Warning: Command failed with return code {result.returncode}")
    return result.returncode == 0


# Task 1 & 2: TP4 vs DP4 Comparison
print("Generating Task 1 & 2 plots (TP4 vs DP4)...")
tp4_file = RUNS_DIR / "losses_large_bs_head8_tp4_rank0.csv"
dp4_file = RUNS_DIR / "losses_large_bs_head8_dp4_rank0.csv"
if tp4_file.exists() and dp4_file.exists():
    run_plot([
        str(tp4_file),
        str(dp4_file),
        "--labels", "TP4", "DP4",
        "--out-step-loss", str(RESULTS_DIR / "task2" / "loss_step.png"),
        "--out-time-loss", str(RESULTS_DIR / "task2" / "loss_time.png"),
        "--out-time-compare", str(RESULTS_DIR / "task2" / "runtime_comparison.png"),
        "--out-mem-avg-bar", str(RESULTS_DIR / "task2" / "memory_comparison.png"),
    ])
else:
    print(f"Warning: Required files not found, skipping Task 1 & 2")

# Task 3 & 4: DP2+TP4 vs DP8 Comparison
print("\nGenerating Task 3 & 4 plots (DP2+TP4 vs DP8)...")
dp2_tp4_file = RUNS_DIR / "losses_large_bs_head8_tp4_dp2_rank0.csv"
dp8_file = RUNS_DIR / "losses_large_bs_head8_dp8_rank0.csv"
if dp2_tp4_file.exists() and dp8_file.exists():
    run_plot([
        str(dp2_tp4_file),
        str(dp8_file),
        "--labels", "DP2+TP4", "DP8",
        "--out-step-loss", str(RESULTS_DIR / "task4" / "loss_step.png"),
        "--out-time-loss", str(RESULTS_DIR / "task4" / "loss_time.png"),
        "--out-time-compare", str(RESULTS_DIR / "task4" / "runtime_comparison.png"),
        "--out-mem-avg-bar", str(RESULTS_DIR / "task4" / "memory_comparison.png"),
    ])
else:
    print(f"Warning: Required files not found, skipping Task 3 & 4")

print("\nDone generating all plots!")

