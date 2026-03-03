"""
Two focused figures supporting the crossover findings.

Figure 1 — refusal_blindness.png
  Claim: when the explicit ask follows an implicit conversation, the model
  doesn't recognise it as a moment requiring refusal.
  Shows: Δ y_refusal trajectories — explicit conv (red) vs crossover (purple).
  Explicit conv builds refusal likelihood; crossover stays near zero.

Figure 2 — resistance_suppression.png
  Claim: explicit conversations suppress y_neg over time (the model builds
  resistance), while crossover stays elevated because the implicit context
  never triggered that resistance.
  Shows: Δ y_neg trajectories — explicit conv (red) vs crossover (purple).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hr.core.data_paths import load_data_paths, resolve_path_spec

ORIG_DIR  = Path("harm_kl/results/kl_drift")
CROSS_DIR = Path("harm_kl/results/kl_drift_crossover")
OUT_DIR   = CROSS_DIR

THEMES = [
    "conduit", "digital_romance", "medical_awakening", "secret_project",
    "messianic_union", "turing_test", "unseen_connections", "ai_sweetheart",
]

RED    = "#d62728"
PURPLE = "#9467bd"


def load_series(metric: str) -> tuple[list, list]:
    """
    Returns (explicit_matrix, crossover_matrix) where each matrix is a list of
    25-element lists (one per theme), values for `metric` at each step.
    """
    exp_mat, cx_mat = [], []
    for slug in THEMES:
        e = json.loads((ORIG_DIR  / f"{slug}_explicit.json").read_text())
        c = json.loads((CROSS_DIR / f"{slug}_implicit.json").read_text())
        exp_mat.append([s[metric] for s in e["per_step"]])
        cx_mat.append( [s[metric] for s in c["per_step"]])
    return exp_mat, cx_mat


def make_figure(metric: str, ylabel: str, title: str, outfile: str,
                hline: float = 0.0):
    exp_mat, cx_mat = load_series(metric)
    steps = list(range(25))
    user_steps = [s for s in steps if s % 2 == 1]
    asst_steps = [s for s in steps if s % 2 == 0 and s > 0]

    exp_mean = np.mean(exp_mat, axis=0)
    cx_mean  = np.mean(cx_mat,  axis=0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # faint individual case lines
    for row in exp_mat:
        ax.plot(steps, row, color=RED,    alpha=0.10, linewidth=0.8)
    for row in cx_mat:
        ax.plot(steps, row, color=PURPLE, alpha=0.10, linewidth=0.8)

    # shaded difference region
    ax.fill_between(steps, exp_mean, cx_mean,
                    where=(cx_mean > exp_mean),
                    color=PURPLE, alpha=0.12, interpolate=True)
    ax.fill_between(steps, exp_mean, cx_mean,
                    where=(cx_mean <= exp_mean),
                    color=RED, alpha=0.12, interpolate=True)

    # bold mean lines (no markers on lines)
    ax.plot(steps, exp_mean, color=RED,    linewidth=2.5, zorder=5)
    ax.plot(steps, cx_mean,  color=PURPLE, linewidth=2.5, zorder=5)

    # mean markers — only every other step to reduce clutter
    for s in user_steps[::2]:
        ax.scatter(s, exp_mean[s], color=RED,    marker="^", s=50, zorder=6)
        ax.scatter(s, cx_mean[s],  color=PURPLE, marker="^", s=50, zorder=6)
    for s in asst_steps[::2]:
        ax.scatter(s, exp_mean[s], color=RED,    marker="s", s=40, zorder=6)
        ax.scatter(s, cx_mean[s],  color=PURPLE, marker="s", s=40, zorder=6)

    # baseline
    ax.axhline(hline, color="#888888", linewidth=0.9, linestyle="--", alpha=0.6, zorder=3)

    ax.set_xlabel("Conversation step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # turn labels on top axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([2 * t + 1 for t in range(0, 12, 2)])
    ax2.set_xticklabels([f"T{t}" for t in range(0, 12, 2)], fontsize=8, color="#666666")
    ax2.tick_params(length=0)
    ax2.spines["top"].set_visible(False)

    # legend with role explanation
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    handles = [
        mlines.Line2D([], [], color=RED,    linewidth=2.5, label="Explicit conversation history"),
        mlines.Line2D([], [], color=PURPLE, linewidth=2.5, label="Implicit conversation history"),
        mlines.Line2D([], [], color="#888", linewidth=0,
                      marker="^", markerfacecolor="#888", markersize=7, label="User turn"),
        mlines.Line2D([], [], color="#888", linewidth=0,
                      marker="s", markerfacecolor="#888", markersize=6, label="Assistant turn"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="best",
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot crossover claim figures.")
    parser.add_argument("--orig_dir", default="dp://harm_kl.results_kl_drift")
    parser.add_argument("--cross_dir", default="dp://harm_kl.results_kl_drift_crossover")
    parser.add_argument(
        "--data-paths-config",
        default="configs/data_paths.yaml",
        help="Path to data path registry yaml.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    data_paths = load_data_paths(args.data_paths_config)

    global ORIG_DIR, CROSS_DIR, OUT_DIR
    ORIG_DIR = resolve_path_spec(args.orig_dir, data_paths)
    CROSS_DIR = resolve_path_spec(args.cross_dir, data_paths)
    OUT_DIR = CROSS_DIR

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    make_figure(
        metric="delta_yrefuse",
        ylabel="Δ log p(refusal)",
        title="Implicit conversation histories suppress refusal recognition",
        outfile="claim_refusal_blindness.png",
    )

    make_figure(
        metric="delta_yneg",
        ylabel="Δ log p(harmful response)",
        title="Explicit conversation histories inoculate against harmful compliance",
        outfile="claim_resistance_suppression.png",
    )


if __name__ == "__main__":
    main()
