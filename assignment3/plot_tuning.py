import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"\[auto-tune\]\s*split_size\s*=\s*(\d+)\s*=>\s*([0-9.]+)\s*([a-zA-Z]+)", re.I)


def parse_log(path: Path):
    d = {}
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PAT.search(ln)
        if not m:
            continue
        size = int(m.group(1))
        val = float(m.group(2))
        unit = m.group(3).lower()
        d[size] = val / 1000.0 if unit == "ms" else val
    if not d:
        return None
    xs = sorted(d)
    ys = [d[x] for x in xs]
    return xs, ys


def plot(xs, ys, title: str, out_png: Path):
    plt.figure(figsize=(7, 4), dpi=160)
    plt.plot(xs, ys, marker="o", lw=2)
    plt.xticks(xs)
    plt.xlabel("Split Size")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.3)
    for x, y in zip(xs, ys):
        plt.annotate(f"{x}", xy=(x, y), xytext=(5, 5),
                     textcoords="offset points", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main(argv):
    if len(argv) < 2:
        sys.exit(f"Usage: {argv[0]} LOG [LOG ...]")
    for s in argv[1:]:
        p = Path(s)
        if not p.exists():
            print(f"[WARN] not found: {p}", file=sys.stderr)
            continue
        parsed = parse_log(p)
        if not parsed:
            print(f"[WARN] no auto-tune lines: {p}", file=sys.stderr)
            continue
        xs, ys = parsed
        out_png = p.with_suffix(".png")
        plot(xs, ys, f"Auto-tune: {p.stem}", out_png)
        print(f"[OK] saved: {out_png}")


if __name__ == "__main__":
    main(sys.argv)
