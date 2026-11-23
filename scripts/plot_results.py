"""
Plot learning curves and diagnostics from metrics.csv files.

usage: python scripts/plot_results.py [--results_dir results/nish] [--out_dir results/nish/plots] [--time wall|train]

Expects <results_dir>/<method>/<task>/<run>/metrics.csv (the layout of logs/<group>/). Writes two figures per actor
delay, each with one panel per task and one line per method:

    return_vs_steps_d<k>.png       eval return vs. environment steps
    return_vs_<time>_time_d<k>.png eval return vs. wall-clock time (or training time with --time train)

A method directory named <method>_d<k> (e.g. baseline_d4) holds runs with actor delay k and goes into the _d<k>
figures; a plain <method> directory is FlashSAC's default delay, d=2. A method has the same color and marker in every
figure, so the d=2 and d=4 figures can be compared side by side. With several runs (seeds) of a method on a task, the
line is their mean and the band spans their min and max. Runs shorter than the longest run of that method and task
are skipped.
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed color and marker per base method, so a method looks the same in every figure. Colors are categorical slots
# 1-8 (light mode, CVD-validated in this order); markers keep methods apart where color alone is not enough.
METHOD_ORDER = ["baseline", "minto", "crossq", "laber"]
LABELS = {"baseline": "Baseline", "minto": "MINTO", "crossq": "Online", "laber": "LaBER"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TASK_ORDER = ["Flat-G1", "Rough-G1", "Flat-H1", "Rough-H1"]

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def short_task(name: str) -> str:
    return name.replace("Isaac-Velocity-", "").replace("-v0", "")


def split_method(method: str) -> tuple[str, Optional[int]]:
    """'baseline_d4' -> ('baseline', 4); 'baseline' -> ('baseline', None)."""
    match = re.fullmatch(r"(.+)_d(\d+)", method)
    return (match.group(1), int(match.group(2))) if match else (method, None)


def label(method: str) -> str:
    base, delay = split_method(method)
    name = LABELS.get(base, base)
    return f"{name} (d={delay})" if delay is not None else name


def read_series(path: str, y_key: str, x_key: str) -> Optional[dict[str, np.ndarray]]:
    """(x, y) for every row of one run where y_key was logged."""
    xs, ys = [], []
    for row in csv.DictReader(open(path)):
        if not row.get(y_key):
            continue
        if x_key == "eval_env_step":  # exact step of the evaluation; the step-0 row has none
            x = float(row.get("eval_env_step") or row["env_step"]) / 1e6
        elif x_key == "env_step":
            x = float(row["env_step"]) / 1e6
        else:  # a time column, in seconds
            x = float(row.get(x_key) or 0.0) / 60.0
        xs.append(x)
        ys.append(float(row[y_key]))
    return {"x": np.array(xs), "y": np.array(ys)} if len(ys) > 1 else None


def load(results_dir: str, y_key: str, x_key: str) -> dict[tuple[str, str], list[dict[str, np.ndarray]]]:
    runs: dict[tuple[str, str], list[dict[str, np.ndarray]]] = defaultdict(list)
    for path in glob.glob(os.path.join(results_dir, "*", "*", "*", "metrics.csv")):
        method, task = path.split(os.sep)[-4], short_task(path.split(os.sep)[-3])
        series = read_series(path, y_key, x_key)
        if series is not None:
            runs[(method, task)].append(series)
    # keep complete runs only, trimmed to a common length so seeds can be averaged point by point
    for key, rs in runs.items():
        n = max(len(r["y"]) for r in rs)
        runs[key] = [r for r in rs if len(r["y"]) >= n - 1]
        n = min(len(r["y"]) for r in runs[key])
        runs[key] = [{"x": r["x"][:n], "y": r["y"][:n]} for r in runs[key]]
    return runs


def by_delay(runs: dict) -> dict[int, dict]:
    """Split runs by actor delay: 'baseline' (d=2, FlashSAC's default) and 'baseline_d4' go to different groups,
    keyed by base method name, so each group is drawn as its own figure with the usual method colors."""
    groups: dict[int, dict] = defaultdict(dict)
    for (method, task), rs in runs.items():
        base, delay = split_method(method)
        groups[delay or 2][(base, task)] = rs
    return groups


def method_sort_key(method: str) -> tuple[int, str, int]:
    base, delay = split_method(method)
    rank = METHOD_ORDER.index(base) if base in METHOD_ORDER else len(METHOD_ORDER)
    return rank, base, delay or 0


def spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Move end-of-line labels apart vertically so they do not overlap; keeps their order."""
    idx = np.argsort(ys)
    placed = np.array(ys, dtype=float)[idx]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    placed -= placed.mean() - np.mean(ys)  # re-center the group on the line ends
    out = np.empty_like(placed)
    out[idx] = placed
    return list(out)


def plot(
    runs, x_label: str, y_label: str, title: str, out_path: str, log_y: bool = False, note: str = ""
) -> None:
    methods = sorted({m for m, _ in runs}, key=method_sort_key)
    tasks = [t for t in TASK_ORDER if any(t == k[1] for k in runs)] + sorted({t for _, t in runs} - set(TASK_ORDER))
    # a method's color comes from its fixed slot, not from which methods happen to be in this figure
    extra = sorted({split_method(m)[0] for m in methods} - set(METHOD_ORDER))
    slots = {b: i for i, b in enumerate(METHOD_ORDER + extra)}
    base_style = {b: (COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]) for b, i in slots.items()}

    ncols = 2 if len(tasks) > 1 else 1
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.0 * nrows), squeeze=False, facecolor=SURFACE)
    seeds = set()

    for ax, task in zip(axes.flat, tasks):
        ax.set_facecolor(SURFACE)
        ends = []
        for method in methods:
            rs = runs.get((method, task))
            if not rs:
                continue
            seeds.add(len(rs))
            color, marker = base_style[split_method(method)[0]]
            x = np.mean([r["x"] for r in rs], axis=0)
            y = np.stack([r["y"] for r in rs])
            mean = y.mean(0)
            every = max(1, len(x) // 10)  # about 10 markers per line, also for the dense training metrics
            ax.plot(x, mean, color=color, lw=2, marker=marker, ms=5, markevery=every, mec=SURFACE, mew=1.2,
                    label=label(method), zorder=3)
            if len(rs) > 1:
                ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=0.15, lw=0, zorder=2)
            ends.append((x[-1], mean[-1], method))

        if log_y:
            ax.set_yscale("log")
        # direct labels at the line ends, in text color (the marker carries the identity)
        if ends:
            ax.autoscale_view()
            y_lo, y_hi = ax.get_ylim()
            to_axis = (lambda v: np.log10(v)) if log_y else (lambda v: v)
            from_axis = (lambda v: 10**v) if log_y else (lambda v: v)
            span = to_axis(y_hi) - to_axis(y_lo)
            label_ys = spread_labels([to_axis(e[1]) for e in ends], min_gap=0.055 * span)
            x_end = max(e[0] for e in ends)
            for (_, y_end, method), y_lab in zip(ends, label_ys):
                value = f"{y_end:.3g}" if log_y else f"{y_end:.2f}" if abs(y_end) < 10 else f"{y_end:.1f}"
                ax.annotate(f"{label(method)} {value}", xy=(x_end, from_axis(y_lab)), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=8, color=TEXT_2,
                            annotation_clip=False)

        ax.set_title(task, color=TEXT, fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel(x_label, color=TEXT_2, fontsize=9)
        ax.set_ylabel(y_label, color=TEXT_2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.8, which="major")
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_2, labelsize=8.5, length=0, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.margins(x=0.02)
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)

    handles, labels = {}, {}
    for ax in axes.flat:  # union over panels, in method order
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    labels = sorted(handles, key=lambda lab: method_sort_key(next(m for m in methods if label(m) == lab)))
    seed_note = "1 seed" if seeds == {1} else "mean, band = min–max over seeds"
    # title and legend sit in a fixed-height band above the panels (in inches, so it works for any number of rows)
    height = fig.get_figheight()
    legend_rows = int(np.ceil(len(labels) / 4))
    band = 0.45 + 0.28 * legend_rows
    fig.tight_layout(rect=(0, 0, 0.93, 1 - band / height))
    fig.suptitle(f"{title} ({seed_note}{'; ' + note if note else ''})", color=TEXT, fontsize=12.5, x=0.01,
                 ha="left", va="top", y=1 - 0.05 / height, fontweight="bold")
    fig.legend([handles[lab] for lab in labels], labels, loc="upper left", ncol=min(len(labels), 4), frameon=False,
               fontsize=9.5, labelcolor=TEXT, bbox_to_anchor=(0.0, 1 - 0.38 / height))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results/nish")
    parser.add_argument("--out_dir", default=None, help="default: <results_dir>/plots")
    parser.add_argument("--time", choices=["wall", "train"], default="wall",
                        help="wall: elapsed time including evaluation; train: training time only (eval excluded)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    time_key = "wall_time" if args.time == "wall" else "eval_train_time"
    time_label = "Wall-clock time (minutes)" if args.time == "wall" else "Training time, excl. evaluation (minutes)"
    steps = "Environment steps (millions)"

    #          name                      y column              x column         y label / title / options
    figures = [
        ("return_vs_steps", "avg_return", "eval_env_step", "Eval return", "Eval return vs. environment steps", {}),
        (f"return_vs_{args.time}_time", "avg_return", time_key, "Eval return", "Eval return vs. wall-clock time",
         {"x_label": time_label}),
    ]
    found = False
    for name, y_key, x_key, y_label, title, opts in figures:
        # one figure per actor delay, so d=2 and d=4 runs are never drawn on the same chart
        for delay, runs in sorted(by_delay(load(args.results_dir, y_key, x_key)).items()):
            found = True
            plot(runs, opts.get("x_label", steps), y_label, title, os.path.join(out_dir, f"{name}_d{delay}.png"),
                 log_y=opts.get("log_y", False), note=f"d={delay}" + (f"; {opts['note']}" if "note" in opts else ""))
    if not found:
        raise SystemExit(f"no metrics.csv found under {args.results_dir}/<method>/<task>/<run>/")


if __name__ == "__main__":
    main()
