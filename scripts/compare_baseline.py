"""
Compare our FlashSAC runs with the paper's results (results/stateRL_gpusim/FlashSAC.csv).

Reads logs/<group>/<exp>/<env>/seed*/metrics.csv (written by CsvTrainerLogger) and writes to --out_dir:
  - <exp>_eval.csv: evaluation returns in the paper's long format (exp_name, env_name, seed, metric, env_step, value),
    so our curves and the paper curves can go through the same plotting code.
  - <exp>_summary.csv: one row per task (final return, normalized AUC, paper numbers, wall-clock, memory).
  - <exp>_curves.png: return vs. env steps (ours per seed + mean, paper mean +- SD).

usage: python scripts/compare_baseline.py [--exp uniform_d2] [--group humanoid] [--out_dir results/ours]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

TASKS = [
    "Isaac-Velocity-Flat-G1-v0",
    "Isaac-Velocity-Rough-G1-v0",
    "Isaac-Velocity-Flat-H1-v0",
    "Isaac-Velocity-Rough-H1-v0",
]
PAPER_CSV = "results/stateRL_gpusim/FlashSAC.csv"
NUM_ENV_STEPS = 50_000_896


def load_run(path: str) -> dict:
    """One metrics.csv -> eval curve and run-level timing numbers."""
    df = pd.read_csv(path)
    seed = int(os.path.basename(os.path.dirname(path)).split("_")[0].removeprefix("seed"))
    evals = df[df["avg_return"].notna()] if "avg_return" in df else df.iloc[0:0]
    # the eval step is logged with each eval; the initial eval (step 0) and older logs only have the logging step
    steps = evals["eval_env_step"].fillna(evals["env_step"]) if "eval_env_step" in evals else evals["env_step"]
    curve = pd.DataFrame({"env_step": steps.values, "avg_return": evals["avg_return"].values})
    curve = curve.drop_duplicates("env_step", keep="last").sort_values("env_step")
    last = df.ffill().iloc[-1]  # last logged value of each column
    return {
        "seed": seed,
        "curve": curve,
        "last_env_step": float(last["env_step"]),
        "complete": float(last["env_step"]) >= NUM_ENV_STEPS - 1024,
        "train_time_h": float(last.get("time/train_time", np.nan)) / 3600,
        "update_time_h": float(last.get("time/update_time", np.nan)) / 3600,
        "collect_time_h": float(last.get("time/collect_time", np.nan)) / 3600,
        "eval_time_h": float(last.get("time/eval_time", np.nan)) / 3600,
        "env_steps_per_sec": float(last.get("time/env_steps_per_sec", np.nan)),
        "gpu_max_memory_gb": float(last.get("time/gpu_max_memory_gb", np.nan)),
    }


def normalized_auc(steps: np.ndarray, returns: np.ndarray, horizon: float = NUM_ENV_STEPS) -> float:
    """Mean return over [0, horizon] (trapezoid rule). Curves that stop early are not extrapolated."""
    if len(steps) < 2:
        return np.nan
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 only has trapz
    # the last eval can land a few interaction steps before the horizon, so accept curves that reach 99% of it
    return float(trapezoid(returns, steps) / steps[-1]) if steps[-1] >= 0.99 * horizon else np.nan


def mean_sd(values: list[float]) -> tuple[float, float, int]:
    v = np.array([x for x in values if np.isfinite(x)])
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="uniform_d2")
    parser.add_argument("--group", default="humanoid")
    parser.add_argument("--log_root", default="logs")
    parser.add_argument("--out_dir", default="results/ours")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paper = pd.read_csv(PAPER_CSV)
    paper = paper[(paper["metric"] == "avg_return") & paper["env_name"].isin(TASKS)]

    long_rows, summary_rows, curves = [], [], {}
    for task in TASKS:
        paths = sorted(glob.glob(os.path.join(args.log_root, args.group, args.exp, task, "seed*", "metrics.csv")))
        runs = [load_run(p) for p in paths]
        # if a seed was run twice, keep the run that got furthest
        by_seed: dict[int, dict] = {}
        for run in runs:
            if run["seed"] not in by_seed or run["last_env_step"] > by_seed[run["seed"]]["last_env_step"]:
                by_seed[run["seed"]] = run
        runs = [by_seed[s] for s in sorted(by_seed)]
        for run in runs:
            for step, ret in run["curve"].itertuples(index=False):
                long_rows.append((args.exp, task, run["seed"], "avg_return", step, ret))

        done = [r for r in runs if r["complete"] and len(r["curve"])]
        ours_final = mean_sd([r["curve"]["avg_return"].iloc[-1] for r in done])
        ours_auc = mean_sd([normalized_auc(r["curve"]["env_step"].values, r["curve"]["avg_return"].values) for r in done])

        p = paper[paper["env_name"] == task]
        p_final, p_auc = [], []
        for _, g in p.groupby("seed"):
            g = g.sort_values("env_step")
            p_final.append(g["value"].iloc[-1])
            p_auc.append(normalized_auc(g["env_step"].values, g["value"].values, horizon=g["env_step"].iloc[-1]))
        paper_final, paper_auc = mean_sd(p_final), mean_sd(p_auc)

        # gap in units of the combined standard error; |z| > 2 means "does not match the paper"
        se = np.sqrt((ours_final[1] ** 2 / max(ours_final[2], 1)) + paper_final[1] ** 2 / paper_final[2])
        summary_rows.append(
            {
                "env_name": task,
                "seeds_done": ours_final[2],
                "seeds_started": len(runs),
                "final_mean": ours_final[0],
                "final_sd": ours_final[1],
                "paper_final_mean": paper_final[0],
                "paper_final_sd": paper_final[1],
                "final_gap_z": (ours_final[0] - paper_final[0]) / se if se > 0 else np.nan,
                "auc_mean": ours_auc[0],
                "auc_sd": ours_auc[1],
                "paper_auc_mean": paper_auc[0],
                "train_time_h": mean_sd([r["train_time_h"] for r in done])[0],
                "update_time_h": mean_sd([r["update_time_h"] for r in done])[0],
                "collect_time_h": mean_sd([r["collect_time_h"] for r in done])[0],
                "eval_time_h": mean_sd([r["eval_time_h"] for r in done])[0],
                "env_steps_per_sec": mean_sd([r["env_steps_per_sec"] for r in done])[0],
                "gpu_max_memory_gb": max([r["gpu_max_memory_gb"] for r in runs], default=np.nan),
            }
        )
        curves[task] = (runs, p)

    pd.DataFrame(long_rows, columns=["exp_name", "env_name", "seed", "metric", "env_step", "value"]).to_csv(
        os.path.join(args.out_dir, f"{args.exp}_eval.csv"), index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.out_dir, f"{args.exp}_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 2):
        print(summary.set_index("env_name").T)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.5 * len(TASKS), 3.6), sharex=True)
    for ax, task in zip(axes, TASKS):
        runs, p = curves[task]
        stats = p.groupby("env_step")["value"].agg(["mean", "std"]).reset_index()
        ax.plot(stats["env_step"] / 1e6, stats["mean"], color="0.3", label="paper (5 seeds)")
        ax.fill_between(
            stats["env_step"] / 1e6, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color="0.3", alpha=0.15
        )
        for run in runs:
            ax.plot(run["curve"]["env_step"] / 1e6, run["curve"]["avg_return"], color="C0", alpha=0.35, lw=1)
        if runs:
            ours = pd.concat([r["curve"] for r in runs]).groupby("env_step")["avg_return"].mean()
            ax.plot(ours.index / 1e6, ours.values, color="C0", lw=2, label=f"ours ({len(runs)} seeds)")
        ax.set_title(task.removeprefix("Isaac-Velocity-").removesuffix("-v0"))
        ax.set_xlabel("env steps (M)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("eval return")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.exp}_curves.png"), dpi=150)
    print(f"wrote {args.out_dir}/{args.exp}_{{eval,summary}}.csv and {args.exp}_curves.png")


if __name__ == "__main__":
    main()
