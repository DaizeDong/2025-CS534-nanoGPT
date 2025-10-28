#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse NanoGPT-style training logs and plot Train/Val loss curves.
Supports lines like:
  iter 4190: loss 0.8694, time 12.06ms, mfu 30.20%
  step 4250: train loss 0.6858, val loss 1.6435

Usage:
  python plot_losses.py /path/to/train.log
  # or read from stdin:
  cat train.log | python plot_losses.py -

Outputs:
  - losses.csv           # parsed values (steps, train_loss, val_loss, iter_loss)
  - loss_plot.png        # loss curves (train vs val)
"""

import sys
import re
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import csv

# ---------- Regex patterns ----------
RE_STEP = re.compile(
    r"""^
        \s*step\s+(\d+)\s*:\s*
        train\s+loss\s+([0-9]*\.?[0-9]+)\s*,\s*
        val\s+loss\s+([0-9]*\.?[0-9]+)
        """,
    re.IGNORECASE | re.VERBOSE,
)

RE_ITER = re.compile(
    r"""^
        \s*iter\s+(\d+)\s*:\s*
        loss\s+([0-9]*\.?[0-9]+)
        (?:,\s*time\s+[0-9]*\.?[0-9]+ms)?
        (?:,\s*mfu\s+[0-9]*\.?[0-9]+%)?
        (?:,\s*mem\s+[0-9]*\.?[0-9]+GB)?
        """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_lines(lines: List[str]):
    steps, train_losses, val_losses = [], [], []
    iters, iter_losses = [], []

    for ln in lines:
        m_step = RE_STEP.search(ln)
        if m_step:
            step = int(m_step.group(1))
            tr = float(m_step.group(2))
            vl = float(m_step.group(3))
            steps.append(step)
            train_losses.append(tr)
            val_losses.append(vl)
            continue

        m_iter = RE_ITER.search(ln)
        if m_iter:
            it = int(m_iter.group(1))
            loss = float(m_iter.group(2))
            iters.append(it)
            iter_losses.append(loss)
            continue

    return (steps, train_losses, val_losses, iters, iter_losses)


def smooth(y: List[float], k: int) -> List[float]:
    """Simple moving average (odd window size)."""
    if not y or k <= 1:
        return y
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    half = k // 2
    out = []
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def write_csv(out_csv: Path, steps, train_losses, val_losses, iters, iter_losses):
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "x", "loss"])
        for x, y in zip(steps, train_losses):
            w.writerow(["train_step", x, y])
        for x, y in zip(steps, val_losses):
            w.writerow(["val_step", x, y])
        for x, y in zip(iters, iter_losses):
            w.writerow(["iter", x, y])


def plot_curves(
    out_png: Path,
    steps, train_losses, val_losses,
    iters, iter_losses,
    iter_smooth: Optional[int] = 0,
):
    plt.figure(figsize=(8, 4.8), dpi=160)

    # Optional: faint per-iter loss as background (smoothed)
    if iters and iter_losses:
        y = iter_losses
        if iter_smooth and iter_smooth > 1:
            y = smooth(y, iter_smooth)
        plt.plot(iters, y, linewidth=1, alpha=0.35, label="Iter Loss")

    # Step-level Train/Val (key curves)
    if steps and train_losses:
        plt.plot(steps, train_losses, marker="o", linewidth=2, label="Train Loss")
    if steps and val_losses:
        plt.plot(steps, val_losses, marker="o", linewidth=2, label="Val Loss")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    # plt.show()  # uncomment for interactive viewing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Path to log file, or '-' to read from stdin.")
    ap.add_argument("--iter-smooth", type=int, default=0,
                    help="Moving average window for per-iter loss (0=off).")
    ap.add_argument("--out-png", type=Path, default=Path("loss_plot.png"))
    ap.add_argument("--out-csv", type=Path, default=Path("losses.csv"))
    args = ap.parse_args()

    # Read input
    if args.logfile == "-":
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(args.logfile).read_text(encoding="utf-8", errors="ignore").splitlines()

    steps, train_losses, val_losses, iters, iter_losses = parse_lines(lines)

    if not steps and not iters:
        sys.stderr.write("No losses found. Make sure log format matches patterns.\n")
        sys.exit(1)

    write_csv(args.out_csv, steps, train_losses, val_losses, iters, iter_losses)
    plot_curves(args.out_png, steps, train_losses, val_losses, iters, iter_losses, args.iter_smooth)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
