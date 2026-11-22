"""Plots for the lab05 report: speedup, efficiency and Karp-Flatt metric.

Input : JSON-lines files written by bnb_tsp.py (one line per repeat).
Output: PNG figures and summary.csv.

Usage:
    python plot_results.py results/bench_n20_2opt.jsonl OUTDIR \
        --ablation results/ablation_none.jsonl results/ablation_minedge.jsonl

Metrics, with T(P) the median time on P workers and the SAME task depth d:
    S(P) = T(1) / T(P)
    E(P) = S(P) / P
    f(P) = (1/S - 1/P) / (1 - 1/P)        (Karp-Flatt, P > 1)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, ScalarFormatter

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 10,
                     "lines.markersize": 5, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

# Curves differ in colour (different luminance), marker and line style, so they
# stay distinguishable in a grayscale print.
COLORS = ["#000000", "#d95f02", "#1b9e77", "#7570b3"]
MARKERS = ["o", "s", "^", "D"]
STYLES = ["-", "--", ":", "-."]

METRICS = [("S", "Speedup S(P)"), ("E", "Efficiency E(P)"),
           ("f", "Karp-Flatt serial fraction f(P)")]


def curve_style(i):
    k = i % len(COLORS)
    return dict(color=COLORS[k], marker=MARKERS[k], ls=STYLES[k])


def load(path):
    """Return the raw runs as a DataFrame (files without 'bound' = none)."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    df = pd.DataFrame(rows)
    if "bound" not in df:
        df["bound"] = "none"
    return df


def summarize(df):
    """Median time per (d, workers); the sequential run is returned separately."""
    seq = df[df.workers == 0]
    par = df[df.workers > 0]
    med = (par.groupby(["d", "workers"])
              .agg(time=("time", "median"), nodes=("nodes", "median"),
                   tasks=("tasks", "first"))
              .reset_index())
    return float(seq.time.median()), float(seq.nodes.median()), med


def add_metrics(med):
    """Add S, E, f relative to the one-worker run with the same d."""
    parts = []
    for _, g in med.groupby("d"):
        base = g[g.workers == 1]
        if base.empty:
            continue
        g = g.copy()
        g["S"] = base.time.iloc[0] / g.time
        g["E"] = g.S / g.workers
        # Karp-Flatt is undefined for P = 1 (division by zero), so skip it there.
        multi = g.workers > 1
        g["f"] = ((1 / g.S - 1 / g.workers) / (1 - 1 / g.workers)).where(multi)
        parts.append(g)
    if not parts:
        raise SystemExit("no depth has a one-worker run to use as the baseline")
    return pd.concat(parts)


def plain_log_ticks(ax, axis, ticks):
    """Log axis (base 2) labelled 1, 2, 4, ... instead of powers."""
    getattr(ax, f"set_{axis}scale")("log", base=2)
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"{axis}axis").set_major_formatter(ScalarFormatter())
    getattr(ax, f"{axis}axis").set_minor_formatter(FuncFormatter(lambda v, _: ""))


def metric_figure(df, group_col, x_col, label_fn, path):
    """Three panels (S, E, f) with one curve per value of `group_col`."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    xs = sorted(df[x_col].unique())
    for ax, (col, ylabel) in zip(axes, METRICS):
        for i, (key, g) in enumerate(df.groupby(group_col)):
            ax.plot(g[x_col], g[col], label=label_fn(key), **curve_style(i))
        if col == "S" and x_col == "workers":
            # Ideal speedup S = P; log-log axes turn it into a diagonal.
            ax.plot(xs, xs, color="0.5", lw=1, label="ideal S = P")
            plain_log_ticks(ax, "x", [p for p in xs if p != 24])  # 24 and 32 would overlap
            plain_log_ticks(ax, "y", [1, 2, 4, 8, 16, 32])
        elif x_col == "workers":
            ax.set_xticks([1, 8, 16, 24, 32, 47])
        else:
            ax.set_xticks(xs)
        if col == "E":
            ax.axhline(1, color="0.5", lw=1)
        if col == "f":
            ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("number of workers P" if x_col == "workers" else "task depth d")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def work_figure(med, seq_nodes, path):
    """Total visited nodes against d, as a multiple of the sequential search."""
    ratio = med.groupby("d").nodes.median() / seq_nodes
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.plot(ratio.index, ratio.values, color="k", marker="o")
    ax.axhline(1, color="0.5", lw=1, label="sequential search = 1")
    for d, v in ratio.items():
        ax.annotate(f"{v:.3g}x", (d, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9)
    if ratio.max() > 3:  # large effect: log axis with plain labels
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_ylim(0.9, 1.1)
    ax.set_xticks(list(ratio.index))
    ax.margins(x=0.2, y=0.25)  # room for the value labels
    ax.set_xlabel("task depth d")
    ax.set_ylabel("visited nodes / sequential")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def true_speedup_figure(med, seq_time, path):
    """Speedup against the plain sequential program, T_seq / T(P)."""
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for i, (d, g) in enumerate(med.groupby("d")):
        ax.plot(g.workers, seq_time / g.time, label=f"d = {d}", **curve_style(i))
    ps = sorted(med.workers.unique())
    ax.plot(ps, ps, color="0.5", lw=1, label="ideal S = P")
    ax.axhline(1, color="0.5", lw=0.8, ls="--")
    plain_log_ticks(ax, "x", [1, 2, 4, 8, 16, 32, 47])
    ax.set_yscale("log", base=2)
    # label values below 1 as fractions (1/4, 1/16, ...) instead of 0.0625
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:g}" if v >= 1 else f"1/{round(1 / v)}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("number of workers P")
    ax.set_ylabel("speedup T_seq / T(P)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ablation_figure(files, path):
    """Visited nodes of the sequential search with and without the bound."""
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for i, f in enumerate(files):
        df = load(f)
        seq = df[df.workers == 0].groupby("n").nodes.median()
        ax.plot(seq.index, seq.values, label=f"bound = {df.bound.iloc[0]}",
                **curve_style(i))
    ax.set_yscale("log")
    ax.set_xlabel("number of cities N")
    ax.set_ylabel("visited nodes (sequential)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data")
    ap.add_argument("outdir")
    ap.add_argument("--ablation", nargs="*", default=[])
    ap.add_argument("--fixed-p", type=int, nargs="+", default=[8, 32],
                    help="worker counts for experiment 2")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4],
                    help="depths drawn in experiment 1")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.data)
    n = int(df.n.iloc[0])
    seq_time, seq_nodes, med = summarize(df)
    metrics = add_metrics(med)
    print(f"N = {n}, sequential time {seq_time:.2f} s, nodes {seq_nodes:.0f}")

    # Experiment 1: fixed d, growing P.
    exp1 = metrics[metrics.d.isin(args.depths)]
    metric_figure(exp1, "d", "workers", lambda d: f"d = {d}",
                  out / "exp1_fixed_d.png")

    # Experiment 2: fixed P, growing d.
    exp2 = metrics[metrics.workers.isin(args.fixed_p)]
    metric_figure(exp2, "workers", "d", lambda p: f"P = {p}",
                  out / "exp2_fixed_p.png")

    work_figure(med, seq_nodes, out / "work_vs_d.png")
    true_speedup_figure(med[med.d.isin(args.depths)], seq_time,
                        out / "speedup_vs_sequential.png")
    if args.ablation:
        ablation_figure(args.ablation, out / "ablation_bound.png")

    metrics.to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
