#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse NanoGPT-style stdout logs and/or enhanced CSV (type,x,loss,elapsed_s,avg_mfu),
and plot *comparative* curves across multiple runs:

  1) Loss vs Step         (per-run Train solid / Val dashed)
  2) Loss vs Elapsed Time (per-run Train solid / Val dashed)
  3) GPU Utilization (MFU %) vs Iteration (distinct color per run)
  4) (optional) Iteration time vs Iteration (overlay per-run)

Key fixes based on feedback:
1) GPU 利用率图的不同曲线使用不同颜色，并按「每个实验/输入」区分。
2) 新增第二张 loss 图：以“运行时间（秒）”为横坐标。
3) 支持对比不同模型/实验：多个输入文件分别绘制为不同曲线（可用 --labels 指定名称）。
   Train 与 Val 线型不同（Train 实线，Val 虚线），颜色随“实验”变化。
4) 避免 step=0 与最后一个 step 被连线：每个实验单独绘制，并在 x 非单调时自动分段绘制。

用法示例：
  # 单个日志
  python plot_metrics.py train.log

  # 多个实验（混合日志与CSV），并命名曲线：
  python plot_metrics.py logs/1gpu.log runs/4gpu/metrics.csv runs/8gpu/metrics.csv \
      --labels "1GPU" "4GPU" "8GPU"

输出：
  loss_step.png   -- Loss vs Step（多实验对比）
  loss_time.png   -- Loss vs Elapsed Time（多实验对比）
  gpu_util.png    -- MFU vs Iteration（多实验对比）
  iter_time.png   -- Iter time vs Iteration（多实验对比；若可推断）
  metrics_parsed.csv -- 统一化明细（含 run 列，便于进一步分析）
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import math
import matplotlib.pyplot as plt

# ---------- Regex (for text logs) ----------
RE_STEP = re.compile(
    r"""^\s*step\s+(\d+)\s*:\s*
        train\s+loss\s+([0-9]*\.?[0-9]+)\s*,\s*
        val\s+loss\s+([0-9]*\.?[0-9]+)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
RE_ITER = re.compile(
    r"""^\s*iter\s+(\d+)\s*:\s*
        loss\s+([0-9]*\.?[0-9]+)
        (?:,\s*time\s+([0-9]*\.?[0-9]+)ms)?
        (?:,\s*mfu\s+([0-9]*\.?[0-9]+)%)?
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------- CSV header detection ----------
CSV_HEADER_MIN = {"type", "x", "loss"}
CSV_HEADER_PLUS = {"elapsed_s", "avg_mfu"}  # optional


def is_csv_file(p: Path) -> bool:
    return p.suffix.lower() == ".csv"


def try_read_csv(path: Path) -> Optional[List[Dict[str, Any]]]:
    """Return list-of-rows if CSV with compatible headers; else None."""
    try:
        with path.open("r", newline="") as f:
            # don't be too fancy: DictReader with default dialect is fine for comma CSV
            reader = csv.DictReader(f)
            headers = set([h.strip() for h in (reader.fieldnames or [])])
            if not CSV_HEADER_MIN.issubset(headers):
                return None
            rows = []
            for row in reader:
                rows.append({(k or "").strip(): (v or "") for k, v in row.items()})
            return rows
    except Exception:
        return None


# ---------- Data containers ----------
class RunData:
    def __init__(self, label: str):
        self.label = label
        # text log
        self.t_steps: List[int] = []
        self.t_train_losses: List[float] = []
        self.t_val_losses: List[float] = []
        self.t_iter_idx: List[int] = []
        self.t_iter_losses: List[float] = []
        self.t_iter_time_ms: List[float] = []
        self.t_iter_mfu: List[float] = []  # percent
        # csv (enhanced)
        self.c_iter_x: List[int] = []
        self.c_iter_loss: List[float] = []
        self.c_iter_elapsed: List[float] = []  # seconds
        self.c_iter_avgmfu: List[float] = []  # percent
        self.c_step_x: List[int] = []
        self.c_train_loss: List[float] = []
        self.c_val_loss: List[float] = []
        self.c_step_elapsed: List[float] = []
        self.c_step_avgmfu: List[float] = []


# ---------- Parsers ----------
def parse_text_log(lines: List[str]):
    it_idx, it_loss, it_time, it_mfu = [], [], [], []
    steps, tr_losses, va_losses = [], [], []
    for ln in lines:
        m = RE_STEP.search(ln)
        if m:
            s = int(m.group(1))
            tr = float(m.group(2))
            va = float(m.group(3))
            steps.append(s);
            tr_losses.append(tr);
            va_losses.append(va)
            continue
        m = RE_ITER.search(ln)
        if m:
            it = int(m.group(1))
            ls = float(m.group(2))
            t_ms = float(m.group(3)) if m.group(3) is not None else float("nan")
            mfu = float(m.group(4)) if m.group(4) is not None else float("nan")
            it_idx.append(it);
            it_loss.append(ls);
            it_time.append(t_ms);
            it_mfu.append(mfu)
            continue
    return steps, tr_losses, va_losses, it_idx, it_loss, it_time, it_mfu


def parse_csv_rows(rows: List[Dict[str, Any]]):
    def _as_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    c_iter_x, c_iter_loss, c_iter_elapsed, c_iter_avgmfu = [], [], [], []
    tr_map: Dict[int, Tuple[float, Optional[float], Optional[float]]] = {}
    va_map: Dict[int, Tuple[float, Optional[float], Optional[float]]] = {}

    for row in rows:
        typ = (row.get("type") or "").strip().lower()
        try:
            x = int(float(row.get("x", "nan")))
        except Exception:
            continue
        loss = _as_float(row.get("loss"))
        elapsed = _as_float(row.get("elapsed_s"))
        avg_mfu = _as_float(row.get("avg_mfu"))

        if typ == "iter":
            if loss is not None:
                c_iter_x.append(x)
                c_iter_loss.append(loss)
                c_iter_elapsed.append(elapsed if elapsed is not None else float("nan"))
                c_iter_avgmfu.append(avg_mfu if avg_mfu is not None else float("nan"))
        elif typ == "train_step":
            if loss is not None:
                tr_map[x] = (loss, elapsed, avg_mfu)
        elif typ == "val_step":
            if loss is not None:
                va_map[x] = (loss, elapsed, avg_mfu)

    c_step_x, c_train_loss, c_val_loss, c_step_elapsed, c_step_avgmfu = [], [], [], [], []
    merged_steps = sorted(set(tr_map.keys()) | set(va_map.keys()))
    for s in merged_steps:
        tr = tr_map.get(s)
        va = va_map.get(s)
        tr_l = tr[0] if tr else float("nan")
        va_l = va[0] if va else float("nan")
        elapsed = (tr[1] if (tr and tr[1] is not None) else (va[1] if (va and va[1] is not None) else float("nan")))
        avg_mfu = (tr[2] if (tr and tr[2] is not None) else (va[2] if (va and va[2] is not None) else float("nan")))
        c_step_x.append(s)
        c_train_loss.append(tr_l)
        c_val_loss.append(va_l)
        c_step_elapsed.append(elapsed)
        c_step_avgmfu.append(avg_mfu)

    return c_iter_x, c_iter_loss, c_iter_elapsed, c_iter_avgmfu, c_step_x, c_train_loss, c_val_loss, c_step_elapsed, c_step_avgmfu


def infer_elapsed_from_iter_time(iter_idx: List[int], iter_time_ms: List[float]) -> Dict[int, float]:
    """Return mapping: iter -> elapsed_s by cumulative sum of per-iter time(ms)."""
    elapsed_map: Dict[int, float] = {}
    acc = 0.0
    last_i = None
    for i, t_ms in sorted(zip(iter_idx, iter_time_ms), key=lambda z: z[0]):
        if (last_i is not None) and (i <= last_i):
            # non-monotonic -> reset accumulation
            acc = 0.0
        if not (t_ms is None or (isinstance(t_ms, float) and math.isnan(t_ms))):
            acc += t_ms / 1000.0
        elapsed_map[i] = acc
        last_i = i
    return elapsed_map


# ---------- Helpers ----------
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
        window = [v for v in y[lo:hi] if not (v is None or (isinstance(v, float) and math.isnan(v)))]
        out.append(sum(window) / len(window) if window else float("nan"))
    return out


def segment_monotonic(x: List[float], y: List[float]) -> List[Tuple[List[float], List[float]]]:
    """
    Split into segments following the ORIGINAL order, breaking whenever:
      - x is non-increasing (xi <= prev_x), or
      - y is NaN/None.
    This avoids connecting step 0 to the last step after a restart and keeps *all* points.
    """
    segs: List[Tuple[List[float], List[float]]] = []
    curx, cury = [], []
    prev = -float("inf")
    for xi, yi in zip(x, y):
        bad_y = (yi is None) or (isinstance(yi, float) and math.isnan(yi))
        if bad_y or xi is None:
            if curx:
                segs.append((curx, cury))
                curx, cury = [], []
            prev = -float("inf")
            continue
        if xi <= prev and curx:
            segs.append((curx, cury))
            curx, cury = [], []
        curx.append(xi);
        cury.append(yi);
        prev = xi
    if curx:
        segs.append((curx, cury))
    return segs


def nan_to_none(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


# ---------- Plotters ----------
def plot_losses_by_step(out_png: Path, runs: List[RunData], labels: List[str]):
    plt.figure(figsize=(8.5, 5.0), dpi=160)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    any_plotted = False
    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        lab = labels[idx]

        # Collect ALL step-level points (keep duplicates, preserve source order)
        train_pairs: List[Tuple[int, float]] = []
        val_pairs: List[Tuple[int, float]] = []

        # from text
        train_pairs += list(zip(run.t_steps, run.t_train_losses))
        val_pairs += list(zip(run.t_steps, run.t_val_losses))

        # from csv (append; do not overwrite)
        for s, tr in zip(run.c_step_x, run.c_train_loss):
            if not (isinstance(tr, float) and math.isnan(tr)):
                train_pairs.append((s, tr))
        for s, va in zip(run.c_step_x, run.c_val_loss):
            if not (isinstance(va, float) and math.isnan(va)):
                val_pairs.append((s, va))

        # Unpack to aligned lists (original order retained)
        train_x = [p[0] for p in train_pairs]
        train_y = [p[1] for p in train_pairs]
        val_x = [p[0] for p in val_pairs]
        val_y = [p[1] for p in val_pairs]

        # Plot ALL segments; label only the first segment per (run, split)
        labeled_train = False
        for xs, ys in segment_monotonic(train_x, train_y):
            plt.plot(
                xs, ys,
                color=color, linewidth=2.0,
                marker='o', markersize=4, markeredgewidth=0.8,
                label=(f"{lab} Train" if not labeled_train else None)
            )
            labeled_train = True
            any_plotted = True

        labeled_val = False
        for xs, ys in segment_monotonic(val_x, val_y):
            plt.plot(
                xs, ys,
                color=color, linewidth=2.0, linestyle='--',
                marker='o', markersize=4, markeredgewidth=0.8, markerfacecolor='white',
                label=(f"{lab} Val" if not labeled_val else None)
            )
            labeled_val = True

    if not any_plotted:
        plt.close()
        return

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss vs Step")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)


def plot_losses_by_time(out_png: Path, runs: List[RunData], labels: List[str]):
    plt.figure(figsize=(8.5, 5.0), dpi=160)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    any_plotted = False

    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        lab = labels[idx]

        # Prefer CSV iter elapsed; else infer from per-iter time
        iter_elapsed_map: Dict[int, float] = {}
        if run.c_iter_x and any([not math.isnan(el) for el in run.c_iter_elapsed]):
            for i, el in zip(run.c_iter_x, run.c_iter_elapsed):
                if not math.isnan(el):
                    iter_elapsed_map[i] = el
        elif run.t_iter_idx and run.t_iter_time_ms:
            iter_elapsed_map = infer_elapsed_from_iter_time(run.t_iter_idx, run.t_iter_time_ms)

        # Collect ALL step-level (elapsed, loss) pairs
        train_pairs: List[Tuple[float, float]] = []
        val_pairs: List[Tuple[float, float]] = []

        # from CSV steps: use provided elapsed if available
        for s, tr, el in zip(run.c_step_x, run.c_train_loss, run.c_step_elapsed):
            if not (isinstance(tr, float) and math.isnan(tr)) and not math.isnan(el):
                train_pairs.append((el, tr))
        for s, va, el in zip(run.c_step_x, run.c_val_loss, run.c_step_elapsed):
            if not (isinstance(va, float) and math.isnan(va)) and not math.isnan(el):
                val_pairs.append((el, va))

        # from TEXT steps: try to map step -> elapsed via iter_elapsed_map
        for s, tr in zip(run.t_steps, run.t_train_losses):
            el = iter_elapsed_map.get(s, None)
            if el is not None:
                train_pairs.append((el, tr))
        for s, va in zip(run.t_steps, run.t_val_losses):
            el = iter_elapsed_map.get(s, None)
            if el is not None:
                val_pairs.append((el, va))

        # Unpack lists (keep original append order)
        train_x = [p[0] for p in train_pairs]
        train_y = [p[1] for p in train_pairs]
        val_x = [p[0] for p in val_pairs]
        val_y = [p[1] for p in val_pairs]

        # Plot ALL segments; label only once per split
        labeled_train = False
        for xs, ys in segment_monotonic(train_x, train_y):
            plt.plot(
                xs, ys,
                color=color, linewidth=2.0,
                marker='o', markersize=4, markeredgewidth=0.8,
                label=(f"{lab} Train" if not labeled_train else None)
            )
            labeled_train = True
            any_plotted = True

        labeled_val = False
        for xs, ys in segment_monotonic(val_x, val_y):
            plt.plot(
                xs, ys,
                color=color, linewidth=2.0, linestyle='--',
                marker='o', markersize=4, markeredgewidth=0.8, markerfacecolor='white',
                label=(f"{lab} Val" if not labeled_val else None)
            )
            labeled_val = True

    if not any_plotted:
        plt.close()
        return

    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss vs Elapsed Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)


def plot_gpu_util(out_png: Path, runs: List[RunData], labels: List[str], util_smooth: int):
    plt.figure(figsize=(8.5, 4.6), dpi=160)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    any_plotted = False

    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        lab = labels[idx]

        # Prefer CSV iter avg MFU; otherwise use text iter MFU
        if run.c_iter_x and any([not math.isnan(v) for v in run.c_iter_avgmfu]):
            x = run.c_iter_x
            y = run.c_iter_avgmfu
            label = f"{lab} (CSV iter avg MFU)"
        elif run.t_iter_idx and any([not math.isnan(v) for v in run.t_iter_mfu]):
            x = run.t_iter_idx
            y = run.t_iter_mfu
            label = f"{lab} (iter MFU)"
        else:
            # nothing to plot for this run
            continue

        if util_smooth and util_smooth > 1:
            y = smooth(y, util_smooth)

        # single colored line per run (确保颜色区分)
        # also add step-level avg_mfu as hollow markers if present
        plt.plot(x, y, color=color, linewidth=2.0, label=label)
        any_plotted = True

        if run.c_step_x and any([not math.isnan(v) for v in run.c_step_avgmfu]):
            xs = [sx for sx, mv in zip(run.c_step_x, run.c_step_avgmfu) if not math.isnan(mv)]
            ys = [mv for mv in run.c_step_avgmfu if not math.isnan(mv)]
            if xs and ys:
                plt.scatter(xs, ys, s=24, facecolors='none', edgecolors=color, label=f"{lab} (step avg MFU)")

    if not any_plotted:
        plt.close()
        return

    plt.xlabel("Iteration")
    plt.ylabel("MFU (%)")
    plt.title("GPU Utilization")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)


def plot_iter_time(out_png: Path, runs: List[RunData], labels: List[str], time_smooth: int):
    plt.figure(figsize=(8.5, 4.6), dpi=160)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    any_plotted = False
    for idx, run in enumerate(runs):
        if not (run.t_iter_idx and run.t_iter_time_ms):
            continue
        color = colors[idx % len(colors)]
        lab = labels[idx]
        x = run.t_iter_idx
        y = run.t_iter_time_ms
        if time_smooth and time_smooth > 1:
            y = smooth(y, time_smooth)
        plt.plot(x, y, color=color, linewidth=1.8, label=f"{lab} Iter time (ms)")
        any_plotted = True
    if not any_plotted:
        plt.close()
        return
    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title("Iteration Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)


# ---------- Unified CSV Writer ----------
def write_unified_csv(out_csv: Path, runs: List[RunData], labels: List[str]):
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "type", "x", "loss", "elapsed_s", "iter_time_ms", "mfu_percent"])

        for lab, run in zip(labels, runs):
            # from text iter
            if run.t_iter_idx:
                # if we can infer elapsed from iter times
                iter_elapsed_map = infer_elapsed_from_iter_time(run.t_iter_idx, run.t_iter_time_ms) if run.t_iter_time_ms else {}
                for i, loss, t_ms, mfu in zip(run.t_iter_idx, run.t_iter_losses, run.t_iter_time_ms, run.t_iter_mfu):
                    w.writerow([lab, "iter", i, float(loss),
                                nan_to_none(iter_elapsed_map.get(i, float("nan"))),
                                nan_to_none(t_ms),
                                nan_to_none(mfu)])

            # from text steps
            for s, tr in zip(run.t_steps, run.t_train_losses):
                w.writerow([lab, "train_step", s, float(tr), "", "", ""])
            for s, va in zip(run.t_steps, run.t_val_losses):
                w.writerow([lab, "val_step", s, float(va), "", "", ""])

            # from csv iter
            for i, loss, el, mfu in zip(run.c_iter_x, run.c_iter_loss, run.c_iter_elapsed, run.c_iter_avgmfu):
                w.writerow([lab, "iter", int(i), float(loss), nan_to_none(el), "", nan_to_none(mfu)])

            # from csv steps
            for s, tr, el, mfu in zip(run.c_step_x, run.c_train_loss, run.c_step_elapsed, run.c_step_avgmfu):
                if not (isinstance(tr, float) and math.isnan(tr)):
                    w.writerow([lab, "train_step", int(s), float(tr), nan_to_none(el), "", nan_to_none(mfu)])
            for s, va, el, mfu in zip(run.c_step_x, run.c_val_loss, run.c_step_elapsed, run.c_step_avgmfu):
                if not (isinstance(va, float) and math.isnan(va)):
                    w.writerow([lab, "val_step", int(s), float(va), nan_to_none(el), "", nan_to_none(mfu)])


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="One or more files: text logs and/or enhanced CSVs.")
    ap.add_argument("--labels", nargs="*", default=None, help="Labels for runs, same length/order as inputs. Defaults to file stems.")
    ap.add_argument("--iter-smooth", type=int, default=0, help="SMA window for per-iter curves (time/MFU).")
    ap.add_argument("--time-smooth", type=int, default=0, help="SMA window for iter time (ms).")
    ap.add_argument("--util-smooth", type=int, default=0, help="SMA window for MFU(%).")
    ap.add_argument("--out-step-loss", type=Path, default=Path("loss_step.png"))
    ap.add_argument("--out-time-loss", type=Path, default=Path("loss_time.png"))
    ap.add_argument("--out-util", type=Path, default=Path("gpu_util.png"))
    ap.add_argument("--out-iter-time", type=Path, default=Path("iter_time.png"))
    ap.add_argument("--out-csv", type=Path, default=Path("metrics_parsed.csv"))
    args = ap.parse_args()

    inputs: List[Path] = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            sys.stderr.write(f"[WARN] file not found: {p}\n")

    # Labels
    if args.labels is not None and len(args.labels) == len(inputs):
        labels = list(args.labels)
    else:
        # fallback: derive from file name
        labels = [p.stem for p in inputs]

    runs: List[RunData] = []

    for lbl, p in zip(labels, inputs):
        run = RunData(lbl)
        if is_csv_file(p):
            rows = try_read_csv(p)
            if rows is None:
                sys.stderr.write(f"[WARN] skip (bad CSV header): {p}\n")
            else:
                (run.c_iter_x, run.c_iter_loss, run.c_iter_elapsed, run.c_iter_avgmfu,
                 run.c_step_x, run.c_train_loss, run.c_val_loss, run.c_step_elapsed, run.c_step_avgmfu) = parse_csv_rows(rows)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            (run.t_steps, run.t_train_losses, run.t_val_losses,
             run.t_iter_idx, run.t_iter_losses, run.t_iter_time_ms, run.t_iter_mfu) = parse_text_log(text)
        runs.append(run)

    # Write unified CSV
    write_unified_csv(args.out_csv, runs, labels)

    # Plots
    plot_losses_by_step(args.out_step_loss, runs, labels)
    plot_losses_by_time(args.out_time_loss, runs, labels)
    plot_gpu_util(args.out_util, runs, labels, args.util_smooth)
    plot_iter_time(args.out_iter_time, runs, labels, args.time_smooth)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_step_loss}")
    print(f"Wrote: {args.out_time_loss}")
    if Path(args.out_util).exists():
        print(f"Wrote: {args.out_util}")
    if Path(args.out_iter_time).exists():
        print(f"Wrote: {args.out_iter_time}")


if __name__ == "__main__":
    main()
