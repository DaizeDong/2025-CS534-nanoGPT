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
        (?:,\s*mem\s+([0-9]*\.?[0-9]+)GB)?
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------- CSV header detection ----------
CSV_HEADER_MIN = {"type", "x", "loss"}
CSV_HEADER_PLUS = {"elapsed_s", "avg_mfu", "mem_gb"}  # optional


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
        self.t_iter_mem_gb: List[float] = []  # GB
        # csv (enhanced)
        self.c_iter_x: List[int] = []
        self.c_iter_loss: List[float] = []
        self.c_iter_elapsed: List[float] = []  # seconds
        self.c_iter_avgmfu: List[float] = []  # percent
        self.c_iter_mem_gb: List[float] = []  # GB
        self.c_step_x: List[int] = []
        self.c_train_loss: List[float] = []
        self.c_val_loss: List[float] = []
        self.c_step_elapsed: List[float] = []
        self.c_step_avgmfu: List[float] = []
        self.c_step_mem_gb: List[float] = []  # GB


# ---------- Parsers ----------
def parse_text_log(lines: List[str]):
    it_idx, it_loss, it_time, it_mfu = [], [], [], []
    it_mem = []
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
            mem = float(m.group(5)) if m.group(5) is not None else float("nan")
            it_idx.append(it);
            it_loss.append(ls);
            it_time.append(t_ms);
            it_mfu.append(mfu)
            it_mem.append(mem)
            continue
    return steps, tr_losses, va_losses, it_idx, it_loss, it_time, it_mfu, it_mem


def parse_csv_rows(rows: List[Dict[str, Any]]):
    def _as_float(v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    c_iter_x, c_iter_loss, c_iter_elapsed, c_iter_avgmfu, c_iter_mem = [], [], [], [], []
    # (loss, elapsed_s, avg_mfu, mem_gb)
    tr_map: Dict[int, Tuple[float, Optional[float], Optional[float], Optional[float]]] = {}
    va_map: Dict[int, Tuple[float, Optional[float], Optional[float], Optional[float]]] = {}

    for row in rows:
        typ = (row.get("type") or "").strip().lower()
        try:
            x = int(float(row.get("x", "nan")))
        except Exception:
            continue
        loss = _as_float(row.get("loss"))
        elapsed = _as_float(row.get("elapsed_s"))
        avg_mfu = _as_float(row.get("avg_mfu"))
        mem_gb = _as_float(row.get("mem_gb"))

        if typ == "iter":
            if loss is not None:
                c_iter_x.append(x)
                c_iter_loss.append(loss)
                c_iter_elapsed.append(elapsed if elapsed is not None else float("nan"))
                c_iter_avgmfu.append(avg_mfu if avg_mfu is not None else float("nan"))
                c_iter_mem.append(mem_gb if mem_gb is not None else float("nan"))
        elif typ == "train_step":
            if loss is not None:
                tr_map[x] = (loss, elapsed, avg_mfu, mem_gb)
        elif typ == "val_step":
            if loss is not None:
                va_map[x] = (loss, elapsed, avg_mfu, mem_gb)

    c_step_x, c_train_loss, c_val_loss, c_step_elapsed, c_step_avgmfu, c_step_mem = [], [], [], [], [], []
    merged_steps = sorted(set(tr_map.keys()) | set(va_map.keys()))
    for s in merged_steps:
        tr = tr_map.get(s)
        va = va_map.get(s)
        tr_l = tr[0] if tr else float("nan")
        va_l = va[0] if va else float("nan")
        elapsed = (tr[1] if (tr and tr[1] is not None) else (va[1] if (va and va[1] is not None) else float("nan")))
        avg_mfu = (tr[2] if (tr and tr[2] is not None) else (va[2] if (va and va[2] is not None) else float("nan")))
        mem_gb = (tr[3] if (tr and tr[3] is not None) else (va[3] if (va and va[3] is not None) else float("nan")))
        c_step_x.append(s)
        c_train_loss.append(tr_l)
        c_val_loss.append(va_l)
        c_step_elapsed.append(elapsed)
        c_step_avgmfu.append(avg_mfu)
        c_step_mem.append(mem_gb)

    return (
        c_iter_x, c_iter_loss, c_iter_elapsed, c_iter_avgmfu, c_iter_mem,
        c_step_x, c_train_loss, c_val_loss, c_step_elapsed, c_step_avgmfu, c_step_mem
    )


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


def shorten_labels(labels: List[str]) -> List[str]:
    """
    Remove common prefix from labels to make them shorter.
    Example: ['large_bs_head8_dp4', 'large_bs_head8_dp8'] -> ['dp4', 'dp8']
    """
    if not labels:
        return labels
    
    if len(labels) == 1:
        return labels
    
    # Find common prefix
    common_prefix = ""
    first_label = labels[0]
    
    for i in range(len(first_label)):
        char = first_label[i]
        # Check if all labels have the same character at this position
        if all(label[i] == char for label in labels if i < len(label)):
            common_prefix += char
        else:
            break
    
    # Only shorten if the common prefix ends with an underscore or is at least 3 characters
    if len(common_prefix) >= 3 and common_prefix.endswith('_'):
        return [label[len(common_prefix):] for label in labels]
    
    return labels


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
    plt.close()


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
    plt.close()


def plot_train_val(out_png: Path, runs: List[RunData], labels: List[str]):
    """Plot train and validation loss on the same figure for a single run."""
    plt.figure(figsize=(8.5, 5.0), dpi=160)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    any_plotted = False
    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        lab = labels[idx]

        # Collect ALL step-level points
        train_pairs: List[Tuple[int, float]] = []
        val_pairs: List[Tuple[int, float]] = []

        # from text
        train_pairs += list(zip(run.t_steps, run.t_train_losses))
        val_pairs += list(zip(run.t_steps, run.t_val_losses))

        # from csv
        for s, tr in zip(run.c_step_x, run.c_train_loss):
            if not (isinstance(tr, float) and math.isnan(tr)):
                train_pairs.append((s, tr))
        for s, va in zip(run.c_step_x, run.c_val_loss):
            if not (isinstance(va, float) and math.isnan(va)):
                val_pairs.append((s, va))

        # Unpack to aligned lists
        train_x = [p[0] for p in train_pairs]
        train_y = [p[1] for p in train_pairs]
        val_x = [p[0] for p in val_pairs]
        val_y = [p[1] for p in val_pairs]

        # Plot train
        for xs, ys in segment_monotonic(train_x, train_y):
            plt.plot(
                xs, ys,
                color=color, linewidth=2.0,
                marker='o', markersize=4, markeredgewidth=0.8,
                label=f"{lab} Train"
            )
            any_plotted = True

        # Plot validation
        for xs, ys in segment_monotonic(val_x, val_y):
            plt.plot(
                xs, ys,
                color=color, linewidth=2.0, linestyle='--',
                marker='o', markersize=4, markeredgewidth=0.8, markerfacecolor='white',
                label=f"{lab} Val"
            )

    if not any_plotted:
        plt.close()
        return

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_final_time_comparison(out_png: Path, runs: List[RunData], labels: List[str]):
    """Plot a bar chart comparing the final elapsed time for each run."""
    plt.figure(figsize=(8.5, 5.0), dpi=160)

    final_times = []
    valid_labels = []

    for idx, run in enumerate(runs):
        # Try to get the last non-NaN elapsed time from step data
        elapsed_times = []

        # From CSV step data
        for el in run.c_step_elapsed:
            if not math.isnan(el):
                elapsed_times.append(el)

        # From CSV iter data
        for el in run.c_iter_elapsed:
            if not math.isnan(el):
                elapsed_times.append(el)

        # From text logs, infer elapsed time
        if run.t_iter_idx and run.t_iter_time_ms:
            iter_elapsed_map = infer_elapsed_from_iter_time(run.t_iter_idx, run.t_iter_time_ms)
            elapsed_times.extend(iter_elapsed_map.values())

        # Get the maximum elapsed time as the final time
        if elapsed_times:
            final_time = max(elapsed_times)
            final_times.append(final_time)
            valid_labels.append(labels[idx])

    if not final_times:
        plt.close()
        return

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bars = plt.bar(range(len(final_times)), final_times, color=[colors[i % len(colors)] for i in range(len(final_times))])

    # Add value labels on top of bars
    for i, (bar, time_val) in enumerate(zip(bars, final_times)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{time_val:.1f}s',
                 ha='center', va='bottom', fontsize=9)

    plt.xlabel("Run")
    plt.ylabel("Final Time (seconds)")
    plt.title("Final Elapsed Time Comparison")
    plt.xticks(range(len(valid_labels)), valid_labels, rotation=45, ha='right')
    plt.grid(True, linestyle="--", alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_gpu_memory_avg_bar(out_png: Path, runs: List[RunData], labels: List[str]):
    """Plot a bar chart comparing the average GPU memory usage for each run."""
    plt.figure(figsize=(8.5, 5.0), dpi=160)

    avg_memories = []
    valid_labels = []

    for idx, run in enumerate(runs):
        # Collect all memory values (excluding NaN)
        memory_values = []

        # From CSV iter data
        for mem in run.c_iter_mem_gb:
            if not math.isnan(mem):
                memory_values.append(mem)

        # From text logs
        for mem in run.t_iter_mem_gb:
            if not math.isnan(mem):
                memory_values.append(mem)

        # Calculate average memory
        if memory_values:
            avg_memory = sum(memory_values) / len(memory_values)
            avg_memories.append(avg_memory)
            valid_labels.append(labels[idx])

    if not avg_memories:
        plt.close()
        return

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bars = plt.bar(range(len(avg_memories)), avg_memories, color=[colors[i % len(colors)] for i in range(len(avg_memories))])

    # Add value labels on top of bars
    for i, (bar, mem_val) in enumerate(zip(bars, avg_memories)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{mem_val:.2f}GB',
                 ha='center', va='bottom', fontsize=9)

    plt.xlabel("Run")
    plt.ylabel("Average GPU Memory (GB)")
    plt.title("Average GPU Memory Usage Comparison")
    plt.xticks(range(len(valid_labels)), valid_labels, rotation=45, ha='right')
    plt.grid(True, linestyle="--", alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="One or more files: text logs and/or enhanced CSVs.")
    ap.add_argument("--labels", nargs="*", default=None, help="Labels for runs, same length/order as inputs. Defaults to file stems.")
    ap.add_argument("--out-step-loss", type=Path, default=Path("loss_step.png"))
    ap.add_argument("--out-time-loss", type=Path, default=Path("loss_time.png"))
    ap.add_argument("--out-train-val", type=Path, default=None)
    ap.add_argument("--out-time-compare", type=Path, default=Path("runtime_comparison.png"))
    ap.add_argument("--out-mem-avg-bar", type=Path, default=Path("memory_comparison.png"))
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
                (run.c_iter_x, run.c_iter_loss, run.c_iter_elapsed,
                 run.c_iter_avgmfu, run.c_iter_mem_gb, run.c_step_x, run.c_train_loss, run.c_val_loss, run.c_step_elapsed, run.c_step_avgmfu, run.c_step_mem_gb) = parse_csv_rows(rows)

        else:
            text = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            (run.t_steps, run.t_train_losses, run.t_val_losses,
             run.t_iter_idx, run.t_iter_losses, run.t_iter_time_ms, run.t_iter_mfu, run.t_iter_mem_gb) = parse_text_log(text)
        runs.append(run)

    # Shorten labels to remove common prefix for better readability
    labels = shorten_labels(labels)

    # Plots
    plot_losses_by_step(args.out_step_loss, runs, labels)
    plot_losses_by_time(args.out_time_loss, runs, labels)
    if args.out_train_val:
        plot_train_val(args.out_train_val, runs, labels)
    plot_final_time_comparison(args.out_time_compare, runs, labels)
    plot_gpu_memory_avg_bar(args.out_mem_avg_bar, runs, labels)

    print(f"Wrote: {args.out_step_loss}")
    print(f"Wrote: {args.out_time_loss}")
    if args.out_train_val and args.out_train_val.exists():
        print(f"Wrote: {args.out_train_val}")
    if args.out_time_compare.exists():
        print(f"Wrote: {args.out_time_compare}")
    if args.out_mem_avg_bar.exists():
        print(f"Wrote: {args.out_mem_avg_bar}")


if __name__ == "__main__":
    main()

