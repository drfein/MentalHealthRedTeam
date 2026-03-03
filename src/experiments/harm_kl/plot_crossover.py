"""
Crossover comparison plots.

The only valid within-message comparison is:
  - Explicit conv  + explicit user_message   (from results/kl_drift/)
  - Implicit conv  + explicit user_message   (from results/kl_drift_crossover/)

Both conditions use the SAME user_message and therefore the SAME log_pi0 baseline,
so Δ values are directly comparable. This isolates the effect of conversation
context (explicit vs implicit history) while holding the probe message constant.

The original implicit results (implicit conv + implicit msg) use a different
baseline and are NOT directly comparable to the explicit condition.

Produces:
  results/kl_drift_crossover/crossover_yneg.png
  results/kl_drift_crossover/crossover_yrefuse.png
  results/kl_drift_crossover/crossover_summary_bar.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.core.data_paths import load_data_paths, resolve_path_spec

ORIG_DIR  = Path("harm_kl/results/kl_drift")
CROSS_DIR = Path("harm_kl/results/kl_drift_crossover")
OUT_DIR   = CROSS_DIR

THEMES = [
    ("conduit",              "The Conduit"),
    ("digital_romance",      "Digital Romance"),
    ("medical_awakening",    "Medical Awakening"),
    ("secret_project",       "Secret Project"),
    ("messianic_union",      "Messianic Union"),
    ("turing_test",          "Turing Test"),
    ("unseen_connections",   "Unseen Connections"),
    ("ai_sweetheart",        "AI Sweetheart"),
]

COLORS = {
    "explicit":   "#d62728",   # red
    "crossover":  "#9467bd",   # purple
}
LABELS = {
    "explicit":  "Explicit conv + explicit msg",
    "crossover": "Implicit conv + explicit msg  ← same baseline, context differs",
}


def load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


import matplotlib.lines as mlines


LEGEND_HANDLES = [
    mlines.Line2D([], [], color=COLORS["explicit"],  linewidth=2.2,
                  label="Explicit conversation history"),
    mlines.Line2D([], [], color=COLORS["crossover"], linewidth=2.2,
                  label="Implicit conversation history"),
    mlines.Line2D([], [], color="#999", linewidth=0,
                  marker="^", markerfacecolor="#999", markersize=6, label="User turn"),
    mlines.Line2D([], [], color="#999", linewidth=0,
                  marker="s", markerfacecolor="#999", markersize=5, label="Assistant turn"),
]


def plot_metric_crossover(metric: str, ylabel: str, title: str, outfile: str):
    ncols = 4
    nrows = (len(THEMES) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.4),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, (slug, label) in enumerate(THEMES):
        ax = axes[i]

        explicit  = load(ORIG_DIR  / f"{slug}_explicit.json")
        crossover = load(CROSS_DIR / f"{slug}_implicit.json")

        all_vals = []
        for key, data in [("explicit", explicit), ("crossover", crossover)]:
            if data is None:
                continue
            steps = [s["step"] for s in data["per_step"]]
            vals  = [s[metric] for s in data["per_step"]]
            roles = [s["role"]  for s in data["per_step"]]
            all_vals.extend(vals)

            ax.plot(steps, vals, color=COLORS[key], linewidth=2.0, alpha=0.9, zorder=3)

            # markers on every other user/assistant step to reduce noise
            for s_val, v, role in zip(steps, vals, roles):
                if s_val % 4 not in (1, 2):
                    continue
                mk = "^" if role == "user" else ("s" if role == "assistant" else None)
                if mk:
                    ax.scatter(s_val, v, color=COLORS[key], marker=mk, s=28, zorder=5)

        # fill between
        if explicit and crossover:
            e_vals = [s[metric] for s in explicit["per_step"]]
            c_vals = [s[metric] for s in crossover["per_step"]]
            steps  = list(range(len(e_vals)))
            ax.fill_between(steps, e_vals, c_vals,
                            where=[c > e for c, e in zip(c_vals, e_vals)],
                            color=COLORS["crossover"], alpha=0.10, interpolate=True)
            ax.fill_between(steps, e_vals, c_vals,
                            where=[c <= e for c, e in zip(c_vals, e_vals)],
                            color=COLORS["explicit"], alpha=0.10, interpolate=True)

        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

        T = (len(explicit["per_step"]) - 1) // 2 if explicit else 12
        ax.set_xlim(0, 2 * T)
        ax.set_title(label, fontsize=9, fontweight="bold", pad=5)
        ax.set_xlabel("Conversation step", fontsize=7, color="#555")
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(labelsize=7, color="#aaa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.legend(handles=LEGEND_HANDLES, loc="lower right",
               fontsize=8.5, framealpha=0.95, edgecolor="#dddddd",
               bbox_to_anchor=(0.98, 0.02))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_summary_bar():
    theme_labels = [label for _, label in THEMES]
    x = np.arange(len(THEMES))
    width = 0.32

    exp_vals, cross_vals = [], []
    for slug, _ in THEMES:
        e = load(ORIG_DIR  / f"{slug}_explicit.json")
        c = load(CROSS_DIR / f"{slug}_implicit.json")
        exp_vals.append(e["per_step"][-1]["delta_yneg"]   if e else 0)
        cross_vals.append(c["per_step"][-1]["delta_yneg"] if c else 0)

    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, exp_vals,   width, color=COLORS["explicit"],  alpha=0.85,
           label="Explicit conversation history")
    ax.bar(x + width / 2, cross_vals, width, color=COLORS["crossover"], alpha=0.85,
           label="Implicit conversation history")

    ax.axhline(0, color="#888", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(theme_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Final Δ log p(harmful response)", fontsize=10)
    ax.set_title("Effect of conversation history on harmful compliance\n"
                 "(both conditions probed with the same explicit user message)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#dddddd")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = OUT_DIR / "crossover_summary_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot KL-drift crossover comparisons.")
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

    plot_metric_crossover(
        metric="delta_yneg",
        ylabel="Δ log p(harmful response)",
        title="Harmful compliance drift by conversation history type",
        outfile="crossover_yneg.png",
    )

    plot_metric_crossover(
        metric="delta_yrefuse",
        ylabel="Δ log p(refusal)",
        title="Refusal recognition by conversation history type",
        outfile="crossover_yrefuse.png",
    )

    plot_summary_bar()


if __name__ == "__main__":
    main()
