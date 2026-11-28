"""
Compare replay-window arms on G1: 10M baseline, fixed 1M buffer (buffer_1M), growing window (window_1to10M).
Reads <results_dir>/<method>/<task>/<run>/metrics.csv (the layout of logs/<group>/ and scripts/plot_results.py).
All 18 runs (3 arms x 2 tasks x 3 seeds) ran on one g6e.xlarge (L40S); the baseline here is our own uniform_d2 sweep.

Metrics per run (normalized AUC = mean return over the range, trapezoid rule; env_step is the logging step):
  AUC 0-15M      window_1to10M runs the same algorithm as buffer_1M here
  AUC 0-20.5M    "early AUC" of the decision rule fixed before the 1M control
  AUC 30-50M     window_1to10M runs the same algorithm as uniform_d2 here
  full AUC, final return (mean of the last two evals)
Writes <out_dir>/three_arm_stats.csv (Welch t per task and pair) and three_arm_G1.png, and prints a mean over the two
tasks of the relative differences (normal approximation; with 3 seeds per arm its CIs are a little too narrow).
No correction for multiple comparisons.

usage: python scripts/compare_arms.py [--results_dir results/jaehoon] [--out_dir results/jaehoon/plots]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

ARMS = {  # key: (exp_name, label, color, linestyle, sampled window in M as a function of env steps in M)
    "B": ("baseline", "10M buffer (baseline)", "#8a8984", "--", lambda x: np.minimum(x, 10)),
    "1M": ("buffer_1M", "1M buffer", "#eb6834", "-", lambda x: np.minimum(x, 1)),
    "W": ("window_1to10M", "growing window 1M→10M", "#2a78d6", "-",
          lambda x: np.minimum(x, np.clip(1 + (x - 15) * 9 / 15, 1, 10))),
}
TASKS = ["Isaac-Velocity-Flat-G1-v0", "Isaac-Velocity-Rough-G1-v0"]
PAIRS = [("W", "B"), ("W", "1M"), ("1M", "B")]
FINAL_STEP = 50_000_896
trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 only has trapz


def auc(s: np.ndarray, r: np.ndarray, lo: float = 0.0, hi: float = np.inf) -> float:
    k = (s >= lo) & (s <= hi)
    s, r = s[k], r[k]
    return float(trapezoid(r, s) / (s[-1] - s[0]))


METRICS = {
    "AUC 0-15M (W = 1M here)": lambda s, r: auc(s, r, hi=15.5e6),
    "early AUC 0-20.5M": lambda s, r: auc(s, r, hi=21e6),
    "late AUC 30-50M (W = B here)": lambda s, r: auc(s, r, lo=30e6),
    "full AUC": lambda s, r: auc(s, r),
    "final return": lambda s, r: float(r[-1]),
}


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h, 2 * stats.t.sf(abs(d / se), df), se


def load_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Eval curve of one run on the logging-step axis (`env_step`), as in scripts/compare_baseline.py."""
    df = pd.read_csv(path)
    e = df[df["avg_return"].notna()][["env_step", "avg_return"]]
    e = e.drop_duplicates("env_step", keep="last").sort_values("env_step")
    return e.env_step.values.astype(float), e.avg_return.values


def load(results_dir: str) -> dict:
    """(task, arm) -> list of (steps, returns), one per seed; checks 3 complete seeds on a shared step grid."""
    runs = {}
    for arm, (method, *_) in ARMS.items():
        for task in TASKS:
            paths = sorted(glob.glob(os.path.join(results_dir, method, task, "seed*", "metrics.csv")))
            R = [load_curve(p) for p in paths]
            assert len(R) == 3 and all(s[-1] == FINAL_STEP for s, _ in R), (task, method, len(R))
            runs[task, arm] = R
    for task in TASKS:
        s0 = runs[task, "B"][0][0]
        assert all(len(s) == len(s0) and (s == s0).all() for a in ARMS for s, _ in runs[task, a]), task
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/jaehoon")
    parser.add_argument("--out_dir", default="results/jaehoon/plots")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = load(args.results_dir)

    rows, rel = [], {(k, x, y): [] for k in METRICS for x, y in PAIRS}
    for task in TASKS:
        print(f"\n== {task[15:-3]}")
        for k, f in METRICS.items():
            v = {a: [f(s, r) for s, r in runs[task, a]] for a in ARMS}
            print(f"  {k:30s} " + "  ".join(f"{a} {np.mean(v[a]):6.2f}±{np.std(v[a], ddof=1):4.2f}" for a in ARMS))
            for x, y in PAIRS:
                d, lo, hi, p, se = welch(v[x], v[y])
                m = np.mean(v[y])
                rel[k, x, y].append((d / m, se / m))
                print(f"      {x:>2} vs {y:<2} {d:+6.2f} ({100 * d / m:+5.1f}%)  "
                      f"CI [{100 * lo / m:+5.1f}%, {100 * hi / m:+5.1f}%]  p={p:.3f}")
                rows.append(dict(task=task[15:-3], metric=k, a=x, b=y, a_mean=np.mean(v[x]), b_mean=m, diff=d,
                                 rel=d / m, rel_lo=lo / m, rel_hi=hi / m, p=p))
            if k.startswith("AUC 0-15M"):  # W and 1M run the same algorithm before 15M: pool them as 6 seeds
                d, lo, hi, p, _ = welch(v["W"] + v["1M"], v["B"])
                m = np.mean(v["B"])
                print(f"      1M+W pooled (n=6) vs B {d:+6.2f} ({100 * d / m:+5.1f}%)  "
                      f"CI [{100 * lo / m:+5.1f}%, {100 * hi / m:+5.1f}%]  p={p:.3f}")
                rows.append(dict(task=task[15:-3], metric=k, a="1M+W pooled", b="B", a_mean=np.mean(v["W"] + v["1M"]),
                                 b_mean=m, diff=d, rel=d / m, rel_lo=lo / m, rel_hi=hi / m, p=p))

    print("\n== mean of the 2 tasks (relative, normal approximation)")
    for k in METRICS:
        for x, y in PAIRS:
            r = rel[k, x, y]
            m, se = np.mean([a for a, _ in r]), np.sqrt(sum(b**2 for _, b in r)) / len(r)
            p = 2 * stats.norm.sf(abs(m / se))
            print(f"  {k:30s} {x:>2} vs {y:<2} {100 * m:+5.1f}%  "
                  f"CI [{100 * (m - 1.96 * se):+5.1f}%, {100 * (m + 1.96 * se):+5.1f}%]  p={p:.3f}")
            rows.append(dict(task="mean of 2", metric=k, a=x, b=y, rel=m, rel_lo=m - 1.96 * se, rel_hi=m + 1.96 * se,
                             p=p))
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "three_arm_stats.csv"), index=False)
    plot(runs, os.path.join(args.out_dir, "three_arm_G1.png"))
    print(f"wrote {args.out_dir}/three_arm_stats.csv and three_arm_G1.png")


def plot(runs: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#2b2b29", "#6b6a65", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.titlecolor": ink, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 5.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
    xs = np.linspace(0, 50, 501)
    for j, task in enumerate(TASKS):
        a, w = ax[0, j], ax[1, j]
        for axis in (a, w):
            axis.axvspan(15, 30, color=ink2, alpha=0.06, lw=0)
        ends = []
        for arm, (_, label, color, ls, window) in ARMS.items():
            s = runs[task, arm][0][0] / 1e6
            Y = np.array([r for _, r in runs[task, arm]])
            m, sd = Y.mean(0), Y.std(0, ddof=1)
            a.fill_between(s, m - sd, m + sd, color=color, alpha=0.14, lw=0)
            a.plot(s, m, color=color, ls=ls, lw=2, marker="o", ms=4, mec="white", mew=1, label=f"{label} (n={len(Y)})")
            ends.append((m[-1], s[-1]))
            w.plot(xs, window(xs), color=color, ls=ls, lw=2)
        ends.sort()  # dodge the end labels so they never overlap
        pos = [ends[0][0]]
        for y, _ in ends[1:]:
            pos.append(max(y, pos[-1] + 1.6))
        for (y, x), py in zip(ends, pos):
            a.annotate(f"{y:.1f}", (x, y), xytext=(x + 0.8, py), va="center", fontsize=8, color=ink2)
        a.set_title(task[15:-3], loc="left", fontsize=10, fontweight="bold")
        a.text(22.5, a.get_ylim()[1] * 0.97, "window grows", va="top", ha="center", fontsize=8, color=ink2)
        a.grid(axis="y", color=grid, lw=0.8)
        a.set_xlim(0, 54)
        w.set_ylim(0, 11)
        w.set_yticks([1, 10])
        w.grid(axis="y", color=grid, lw=0.8)
        w.set_xlabel("env steps (M)")
    ax[0, 0].set_ylabel("eval return\n(mean ± 1 SD over seeds)")
    ax[1, 0].set_ylabel("sampled window\n(M transitions)")
    ax[0, 0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("FlashSAC on G1: replay window size, same code and GPU (L40S), 3 seeds per arm",
                 x=0.01, ha="left", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")


if __name__ == "__main__":
    main()
