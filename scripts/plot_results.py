"""
Plot learning curves and diagnostics from metrics.csv files.

usage: python scripts/plot_results.py [--results_dir results/nish [more dirs]] [--out_dir results/nish/plots]
                                      [--time wall|train]
                                      [--methods crossq crossq_1q ...] [--suffix _critic]

Expects <results_dir>/<method>/<task>/<run>/metrics.csv (the layout of logs/<group>/). Writes two figures per actor
delay, each with one panel per task and one line per method:

    return_vs_steps_d<k>.png       eval return vs. environment steps
    return_vs_<time>_time_d<k>.png eval return vs. wall-clock time (or training time with --time train)

plus, for the d=2 methods, the normalized area under the return curve (auc.png and auc.csv): the mean return over
0-50M steps, and the mean return over a common time budget per task (the shortest final time of any run on it).
The wall-clock figure labels each line "<method> (<final return>, <mean final time> min)".

paper_vs_baseline.png / .csv compare our baseline (--paper_method) with the FlashSAC paper (--paper_csv): final return
and AUC, ours - paper, in % and as z = difference / combined standard error.

Step axis: every step-based figure and AUC uses the logging step (`env_step`), where each eval is recorded at the next
logging step (the 5.00M eval at 5.50M, ...). The FlashSAC paper's CSV and scripts/compare_baseline.py use the same
axis, so all AUCs here are directly comparable with the paper's and the team's. (`eval_env_step`, the exact step of
an eval, would shift the curves ~0.5M steps left and raise every AUC by ~0.35.)

Asymmetric-critic runs: a method named <method>_asym (e.g. crossq_1q_w360_asym) is drawn in the color and marker of
<method>, with a dashed line and a hollow marker, and labeled "<method label> + asym. critic".

Rafay's runs, compared with our baseline:
    python scripts/plot_results.py --results_dir results/rafay results/nish \
        --methods baseline architecture_linear_bottleneck architecture_swiglu --out_dir results/rafay/plots

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
METHOD_ORDER = ["baseline", "minto", "crossq", "laber", "crossq_1q", "crossq_1q_w360", "crossq_sw", "crossq_1q_w360_sw",
                "architecture_linear_bottleneck", "architecture_swiglu"]
LABELS = {
    "baseline": "Baseline",
    "minto": "MINTO",
    "crossq": "Online",
    "laber": "LaBER",
    "crossq_1q": "Online, 1 critic",
    "crossq_1q_w360": "Online, 1 wide critic",
    "crossq_sw": "Online + surprise wt.",
    "crossq_1q_w360_sw": "Online, 1 wide critic + surprise wt.",
    "architecture_linear_bottleneck": "Linear bottleneck",
    "architecture_swiglu": "SwiGLU",
    # asymmetric-critic variants (run_humanoid.sh --asym); any other <method>_asym gets "<label> + asym. critic"
    "crossq_asym": "Online + asym. critic",
    "crossq_1q_asym": "Online, 1 critic + asym. critic",
    "crossq_1q_w360_asym": "Online, 1 wide critic + asym. critic",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
MARKER_SCALE = {"*": 1.8}  # the star draws much smaller than the other markers at the same size
# There are 8 color slots and more methods, so Rafay's two block architectures reuse the slots of two archived methods
# (laber, crossq_1q_w360_sw; now in results/nish/old_runs) that are never drawn in the same figure as them.
# Baseline + linear bottleneck + SwiGLU (slots 1, 4, 8) passes the colorblind-safety validator.
SLOTS = {m: i for i, m in enumerate(METHOD_ORDER[:8])} | {"architecture_linear_bottleneck": 3, "architecture_swiglu": 7}


ASYM_SUFFIX = "_asym"  # asymmetric-critic variant of a method (run_humanoid.sh --asym)


def asym_partner(base: str) -> tuple[str, bool]:
    """'crossq_1q_asym' -> ('crossq_1q', True); 'crossq_1q' -> ('crossq_1q', False)."""
    return (base[: -len(ASYM_SUFFIX)], True) if base.endswith(ASYM_SUFFIX) else (base, False)


def style(base: str) -> tuple[str, str]:
    """(color, marker) of a base method: its fixed slot, or the next free slot for a method not listed above.
    An asymmetric variant shares its partner's slot; line_style() and hollow() tell the two apart."""
    base = asym_partner(base)[0]
    if base not in SLOTS:
        SLOTS[base] = len(SLOTS) % len(COLORS)
    return COLORS[SLOTS[base]], MARKERS[SLOTS[base] % len(MARKERS)]


def is_asym(method: str) -> bool:
    return asym_partner(split_method(method)[0])[1]


def line_style(method: str) -> str:
    return "--" if is_asym(method) else "-"
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
    partner, asym = asym_partner(base)
    name = LABELS.get(base) or (f"{LABELS.get(partner, partner)} + asym. critic" if asym else base)
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


def load(results_dirs: list[str], y_key: str, x_key: str) -> dict[tuple[str, str], list[dict[str, np.ndarray]]]:
    runs: dict[tuple[str, str], list[dict[str, np.ndarray]]] = defaultdict(list)
    paths = [p for d in results_dirs for p in glob.glob(os.path.join(d, "*", "*", "*", "metrics.csv"))]
    for path in paths:
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


def method_sort_key(method: str) -> tuple[int, int, str, int]:
    """Order of the methods in legends and tables; an asymmetric variant comes right after its partner."""
    base, delay = split_method(method)
    partner, asym = asym_partner(base)
    rank = METHOD_ORDER.index(partner) if partner in METHOD_ORDER else len(METHOD_ORDER)
    return rank, int(asym), partner, delay or 0


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
    runs, x_label: str, y_label: str, title: str, out_path: str, log_y: bool = False, note: str = "",
    x_in_label: str = "",
) -> None:
    """x_in_label: if set (a unit, e.g. 'min'), each end-of-line label also shows the mean final x of that method."""
    methods = sorted({m for m, _ in runs}, key=method_sort_key)
    tasks = [t for t in TASK_ORDER if any(t == k[1] for k in runs)] + sorted({t for _, t in runs} - set(TASK_ORDER))
    # a method's color comes from its fixed slot, not from which methods happen to be in this figure
    base_style = {b: style(b) for b in sorted({split_method(m)[0] for m in methods}, key=method_sort_key)}

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
            hollow = is_asym(method)  # asymmetric variant: dashed line, hollow marker in the partner's color
            ax.plot(x, mean, color=color, lw=2, ls=line_style(method), marker=marker,
                    ms=5 * MARKER_SCALE.get(marker, 1), markevery=every, mfc=SURFACE if hollow else color,
                    mec=color if hollow else SURFACE, mew=1.4 if hollow else 1.2,
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
            for (x_last, y_end, method), y_lab in zip(ends, label_ys):
                value = f"{y_end:.3g}" if log_y else f"{y_end:.2f}" if abs(y_end) < 10 else f"{y_end:.1f}"
                text = f"{label(method)} ({value}, {x_last:.1f} {x_in_label})" if x_in_label else f"{label(method)} {value}"
                ax.annotate(text, xy=(x_end, from_axis(y_lab)), xytext=(8, 0),
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


def mean_return_until(x: np.ndarray, y: np.ndarray, budget: float) -> float:
    """Mean of the return curve over [0, budget] (trapezoid rule), interpolating the curve at the budget."""
    keep = x < budget
    xs = np.append(x[keep], budget)
    ys = np.append(y[keep], np.interp(budget, x, y))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 only has trapz
    return float(trapezoid(ys, xs) / budget)


def auc_table(results_dir: list[str], time_key: str, methods: Optional[list[str]]) -> list[dict]:
    """Normalized area under the eval-return curve per method and task, two ways:

    steps: mean return over [0, final step], on the logging-step axis. Same as scripts/compare_baseline.py.
    time:  mean return over [0, T], with T the shortest final time of any run of that task, so every method gets the
           same time budget and no curve is extrapolated.
    """
    steps = {k: v for k, v in load(results_dir, "avg_return", "env_step").items() if split_method(k[0])[1] is None}
    times = {k: v for k, v in load(results_dir, "avg_return", time_key).items() if split_method(k[0])[1] is None}
    if methods is not None:
        steps = {k: v for k, v in steps.items() if k[0] in methods}
        times = {k: v for k, v in times.items() if k[0] in methods}
    rows = []
    for task in sorted({t for _, t in steps}, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
        budget = min(r["x"][-1] for (m, t), rs in times.items() if t == task for r in rs)
        for method in sorted({m for m, t in steps if t == task}, key=method_sort_key):
            s = [mean_return_until(r["x"], r["y"], r["x"][-1]) for r in steps[(method, task)]]
            w = [mean_return_until(r["x"], r["y"], budget) for r in times[(method, task)]]
            rows.append({"method": method, "task": task, "seeds": len(s),
                         "auc_steps": float(np.mean(s)), "auc_steps_sd": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
                         "auc_steps_seeds": s, "auc_time": float(np.mean(w)),
                         "auc_time_sd": float(np.std(w, ddof=1)) if len(w) > 1 else 0.0, "auc_time_seeds": w,
                         "time_budget_min": budget})
    for row in rows:  # change relative to the baseline on the same task
        base = next((b for b in rows if b["task"] == row["task"] and b["method"] == "baseline"), None)
        for k in ("auc_steps", "auc_time"):
            row[f"{k}_vs_baseline_pct"] = 100 * (row[k] / base[k] - 1) if base else float("nan")
    return rows


def write_auc_csv(rows: list[dict], out_path: str) -> None:
    cols = ["method", "task", "seeds", "auc_steps", "auc_steps_sd", "auc_steps_vs_baseline_pct", "auc_time",
            "auc_time_sd", "auc_time_vs_baseline_pct", "time_budget_min"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols + ["auc_steps_per_seed", "auc_time_per_seed"])
        for r in rows:
            writer.writerow([r[c] if isinstance(r[c], str) else round(r[c], 4) for c in cols]
                            + [" ".join(f"{v:.3f}" for v in r["auc_steps_seeds"]),
                               " ".join(f"{v:.3f}" for v in r["auc_time_seeds"])])
    print(f"saved {out_path}")


def plot_auc(rows: list[dict], time_label: str, out_path: str) -> None:
    """Dot plot: one panel per (AUC kind, task); a row per method with its seeds (small dots) and mean (large marker).
    Dots, not bars: the differences are a few percent of the value, which bars from zero would hide."""
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in rows)]
    methods = sorted({r["method"] for r in rows}, key=method_sort_key)
    kinds = [("auc_steps", "Mean return over 0–50M steps"), ("auc_time", f"Mean return over a common {time_label}")]

    fig, axes = plt.subplots(len(kinds), len(tasks), figsize=(3.6 * len(tasks), 0.42 * len(methods) * 2 + 2.6),
                             squeeze=False, facecolor=SURFACE)
    for i, (key, kind_title) in enumerate(kinds):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            task_rows = {r["method"]: r for r in rows if r["task"] == task}
            for k, method in enumerate(methods):
                r = task_rows.get(method)
                if r is None:
                    continue
                color, marker = style(split_method(method)[0])
                yk = len(methods) - 1 - k
                ax.scatter(r[f"{key}_seeds"], [yk] * len(r[f"{key}_seeds"]), s=22, color=color, alpha=0.45, lw=0,
                           zorder=2)
                hollow = is_asym(method)
                ax.scatter([r[key]], [yk], s=70 * MARKER_SCALE.get(marker, 1) ** 2, marker=marker,
                           facecolors=SURFACE if hollow else color, edgecolors=color if hollow else SURFACE,
                           linewidths=1.8 if hollow else 1.5,
                           zorder=3)
                pct = r[f"{key}_vs_baseline_pct"]
                note = "" if method == "baseline" or not np.isfinite(pct) else f"  {pct:+.1f}%"
                ax.annotate(f"{r[key]:.1f}{note}", xy=(r[key], yk), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8, color=TEXT_2)
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([label(m) for m in reversed(methods)] if j == 0 else [], fontsize=8.5, color=TEXT)
            ax.set_ylim(-0.6, len(methods) - 0.3)
            budget = next(r["time_budget_min"] for r in rows if r["task"] == task)
            ax.set_title(task + (f"  (first {budget:.0f} min)" if key == "auc_time" else ""), color=TEXT,
                         fontsize=10.5, loc="left", fontweight="bold")
            if j == 0:
                ax.text(0, 1.22, kind_title, transform=ax.transAxes, fontsize=10, color=TEXT_2, va="bottom")
            ax.grid(True, axis="x", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    seeds = sorted({r["seeds"] for r in rows})
    fig.suptitle(f"Area under the eval-return curve (large marker = mean of {'/'.join(map(str, seeds))} seeds, "
                 f"small dots = seeds; % vs. baseline)", color=TEXT, fontsize=12, x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


PAPER_CSV = "results/stateRL_gpusim/FlashSAC.csv"


def load_paper(path: str) -> dict[str, list[dict[str, np.ndarray]]]:
    """The FlashSAC paper's eval-return curves, one per seed, keyed by short task name. The paper records each eval at
    the next logging step (the 5.00M eval at 5.50M, ...), the same axis as the `env_step` column of our rows."""
    by_seed: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in csv.DictReader(open(path)):
        task = short_task(row["env_name"])
        if row["metric"] == "avg_return" and task in TASK_ORDER:
            by_seed[(task, row["seed"])].append((float(row["env_step"]) / 1e6, float(row["value"])))
    curves: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    for (task, _), points in sorted(by_seed.items()):
        x, y = zip(*sorted(points))
        curves[task].append({"x": np.array(x), "y": np.array(y)})
    return curves


def paper_comparison(ours: dict, paper: dict, method: str) -> list[dict]:
    """Final return and normalized AUC of our `method` vs. the paper, per task, on the paper's step axis.
    Same definitions as scripts/compare_baseline.py: final = last eval; AUC = trapezoid area / last step;
    z = (ours - paper) / combined standard error of the two means."""

    def stats(values: list[float]) -> tuple[float, float, int]:
        return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, len(values)

    rows = []
    for task in TASK_ORDER:
        if (method, task) not in ours or task not in paper:
            continue
        row: dict = {"task": task}
        for who, runs in (("ours", ours[(method, task)]), ("paper", paper[task])):
            row[f"{who}_final"], row[f"{who}_final_sd"], row[f"{who}_seeds"] = stats([r["y"][-1] for r in runs])
            row[f"{who}_auc"], row[f"{who}_auc_sd"], _ = stats(
                [mean_return_until(r["x"], r["y"], r["x"][-1]) for r in runs])
        for k in ("final", "auc"):
            diff = row[f"ours_{k}"] - row[f"paper_{k}"]
            se = np.sqrt(row[f"ours_{k}_sd"] ** 2 / row["ours_seeds"] + row[f"paper_{k}_sd"] ** 2 / row["paper_seeds"])
            row[f"{k}_diff"], row[f"{k}_diff_pct"] = diff, 100 * diff / row[f"paper_{k}"]
            row[f"{k}_z"] = diff / se if se > 0 else float("nan")
        rows.append(row)
    return rows


def write_paper_csv(rows: list[dict], out_path: str) -> None:
    cols = ["task", "ours_seeds", "paper_seeds", "ours_final", "ours_final_sd", "paper_final", "paper_final_sd",
            "final_diff", "final_diff_pct", "final_z", "ours_auc", "ours_auc_sd", "paper_auc", "paper_auc_sd",
            "auc_diff", "auc_diff_pct", "auc_z"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([r[c] if isinstance(r[c], (str, int)) else round(r[c], 4) for c in cols])
    print(f"saved {out_path}")


def plot_paper(rows: list[dict], ours: dict, paper: dict, method: str, out_path: str) -> None:
    """Our `method` vs. the paper per task: mean curve with a min–max band over seeds, the paper as a dashed
    reference line, and the differences in each panel title."""
    color, marker = style(split_method(method)[0])
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), squeeze=False, facecolor=SURFACE)
    for ax, row in zip(axes.flat, rows):
        task = row["task"]
        ax.set_facecolor(SURFACE)
        ends = []
        for runs, name, c, ls, mk in ((paper[task], f"Paper ({row['paper_seeds']} seeds)", TEXT_2, "--", None),
                                      (ours[(method, task)], f"Ours, {label(method)} ({row['ours_seeds']} seeds)",
                                       color, "-", marker)):
            n = min(len(r["y"]) for r in runs)
            x = np.mean([r["x"][:n] for r in runs], axis=0)
            y = np.stack([r["y"][:n] for r in runs])
            ax.plot(x, y.mean(0), color=c, lw=2, ls=ls, marker=mk, ms=5, mec=SURFACE, mew=1.2, label=name, zorder=3)
            ax.fill_between(x, y.min(0), y.max(0), color=c, alpha=0.12, lw=0, zorder=2)
            ends.append((x[-1], y.mean(0)[-1], name.split(" (")[0]))
        ax.autoscale_view()
        y_lo, y_hi = ax.get_ylim()
        for (x_end, y_end, name), y_lab in zip(ends, spread_labels([e[1] for e in ends], 0.06 * (y_hi - y_lo))):
            ax.annotate(f"{name} {y_end:.1f}", xy=(x_end, y_lab), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT_2, annotation_clip=False)
        ax.set_title(task, color=TEXT, fontsize=11, loc="left", fontweight="bold", pad=20)
        ax.text(0, 1.02, f"final {row['final_diff']:+.2f} ({row['final_diff_pct']:+.1f}%, z {row['final_z']:+.1f})   "
                f"AUC {row['auc_diff']:+.2f} ({row['auc_diff_pct']:+.1f}%, z {row['auc_z']:+.1f})",
                transform=ax.transAxes, fontsize=8.5, color=TEXT_2, va="bottom", ha="left")
        ax.set_xlabel("Environment steps (millions), logging step as in the paper", color=TEXT_2, fontsize=9)
        ax.set_ylabel("Eval return", color=TEXT_2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_2, labelsize=8.5, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.margins(x=0.02)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 0.92, 0.88), h_pad=2.5)
    fig.suptitle(f"Ours vs. the FlashSAC paper (mean, band = min–max over seeds; ours − paper above each panel)",
                 color=TEXT, fontsize=12.5, x=0.01, ha="left", y=0.985, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, fontsize=9.5, labelcolor=TEXT,
               bbox_to_anchor=(0.0, 0.955))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", nargs="+", default=["results/nish"],
                        help="one or more folders; e.g. results/rafay results/nish to compare against our baseline")
    parser.add_argument("--out_dir", default=None, help="default: <first results_dir>/plots")
    parser.add_argument("--time", choices=["wall", "train"], default="wall",
                        help="wall: elapsed time including evaluation; train: training time only (eval excluded)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="only plot these method directories, e.g. --methods crossq crossq_1q crossq_1q_w360")
    parser.add_argument("--suffix", default="", help="appended to the figure file names, e.g. _critic")
    parser.add_argument("--paper_csv", default=PAPER_CSV, help="the FlashSAC paper's results, for paper_vs_<method>")
    parser.add_argument("--paper_method", default="baseline", help="which of our methods to compare with the paper")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir[0], "plots")
    os.makedirs(out_dir, exist_ok=True)
    time_key = "wall_time" if args.time == "wall" else "eval_train_time"
    time_label = "Wall-clock time (minutes)" if args.time == "wall" else "Training time, excl. evaluation (minutes)"
    steps = "Environment steps (millions)"

    #          name                      y column              x column         y label / title / options
    figures = [
        ("return_vs_steps", "avg_return", "env_step", "Eval return", "Eval return vs. environment steps", {}),
        (f"return_vs_{args.time}_time", "avg_return", time_key, "Eval return", "Eval return vs. wall-clock time",
         {"x_label": time_label, "x_in_label": "min"}),
    ]
    found = False
    for name, y_key, x_key, y_label, title, opts in figures:
        # one figure per actor delay, so d=2 and d=4 runs are never drawn on the same chart
        loaded = load(args.results_dir, y_key, x_key)
        if args.methods is not None:
            loaded = {key: value for key, value in loaded.items() if key[0] in args.methods}
        for delay, runs in sorted(by_delay(loaded).items()):
            found = True
            out_path = os.path.join(out_dir, f"{name}_d{delay}{args.suffix}.png")
            plot(runs, opts.get("x_label", steps), y_label, title, out_path, x_in_label=opts.get("x_in_label", ""),
                 log_y=opts.get("log_y", False), note=f"d={delay}" + (f"; {opts['note']}" if "note" in opts else ""))
    if not found:
        raise SystemExit(f"no metrics.csv found under {' '.join(args.results_dir)} (<dir>/<method>/<task>/<run>/)")

    # area under the curve, for the d=2 methods (vs. steps and vs. a common time budget per task)
    rows = auc_table(args.results_dir, time_key, args.methods)
    if rows:
        budget_label = "wall-clock budget" if args.time == "wall" else "training-time budget"
        write_auc_csv(rows, os.path.join(out_dir, f"auc{args.suffix}.csv"))
        plot_auc(rows, budget_label, os.path.join(out_dir, f"auc{args.suffix}.png"))

    # our baseline vs. the paper, on the paper's step axis (the logging step)
    if os.path.exists(args.paper_csv) and (args.methods is None or args.paper_method in args.methods):
        ours = load(args.results_dir, "avg_return", "env_step")
        rows = paper_comparison(ours, load_paper(args.paper_csv), args.paper_method)
        if rows:
            name = f"paper_vs_{args.paper_method}{args.suffix}"
            write_paper_csv(rows, os.path.join(out_dir, f"{name}.csv"))
            plot_paper(rows, ours, load_paper(args.paper_csv), args.paper_method, os.path.join(out_dir, f"{name}.png"))


if __name__ == "__main__":
    main()
