"""
Plot per-turn KL-drift results for all cases.
Produces two figures saved to results/kl_drift/:
  - harm_amplification.png  : grid of harm_amplification trajectories
  - pref_prob.png           : grid of p(y+ > y-) trajectories
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

RESULTS_DIR = Path("harm_kl/results/kl_drift")
OUT_DIR = RESULTS_DIR

CASE_ORDER = [
    "conduit_explicit", "conduit_implicit",
    "digital_romance_explicit", "digital_romance_implicit",
    "medical_awakening_explicit", "medical_awakening_implicit",
    "secret_project_explicit", "secret_project_implicit",
    "messianic_union_explicit", "messianic_union_implicit",
    "turing_test_explicit", "turing_test_implicit",
    "unseen_connections_explicit", "unseen_connections_implicit",
    "ai_sweetheart_explicit", "ai_sweetheart_implicit",
]

CONDITION_COLOR = {"Explicit": "#d62728", "Implicit": "#1f77b4"}


def load_results() -> list[dict]:
    results = {}
    for case_id in CASE_ORDER:
        path = RESULTS_DIR / f"{case_id}.json"
        if path.exists():
            results[case_id] = json.loads(path.read_text())
    return [results[k] for k in CASE_ORDER if k in results]


def short_title(name: str) -> str:
    # "Case 1.1: The Conduit [EXPLICIT]" → "1.1 The Conduit"
    parts = name.replace("[EXPLICIT]", "").replace("[IMPLICIT]", "").strip()
    parts = parts.replace("Case ", "")
    num, rest = parts.split(":", 1)
    return f"{num.strip()}: {rest.strip()}"


USER_COLOR      = "#e07b00"   # orange for user steps
ASSISTANT_COLOR = "#5b5ea6"   # purple for assistant steps


def plot_metric(results, metric: str, ylabel: str, title: str, outfile: str,
                hline: float | None = None, hline_label: str = ""):
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.4),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, d in enumerate(results):
        ax = axes[i]
        steps  = [s["step"]  for s in d["per_step"]]
        values = [s[metric]  for s in d["per_step"]]
        roles  = [s["role"]  for s in d["per_step"]]

        condition = d.get("condition", "Explicit")
        line_color = CONDITION_COLOR.get(condition, "#333333")

        ax.plot(steps, values, color=line_color, linewidth=1.5,
                zorder=2, alpha=0.7)

        # scatter points coloured by role
        for s_val, v, role in zip(steps, values, roles):
            if role == "user":
                c, mk = USER_COLOR, "^"
            elif role == "assistant":
                c, mk = ASSISTANT_COLOR, "s"
            else:
                c, mk = "black", "o"
            ax.scatter(s_val, v, color=c, marker=mk, s=28, zorder=4)

        baseline = hline if hline is not None else 0
        ax.fill_between(steps, values, baseline,
                        where=[v > baseline for v in values],
                        alpha=0.12, color="#d62728", interpolate=True)
        ax.fill_between(steps, values, baseline,
                        where=[v <= baseline for v in values],
                        alpha=0.12, color="#1f77b4", interpolate=True)

        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.8, linestyle="--",
                       alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        # light vertical bands to delineate turns
        for t in range(d["T"]):
            ax.axvspan(2 * t + 1, 2 * t + 2, alpha=0.04, color="gray")

        ax.set_title(short_title(d["name"]), fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Step (▲=user  ■=assistant)", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 2 * d["T"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=CONDITION_COLOR["Explicit"],  label="Explicit"),
        mpatches.Patch(color=CONDITION_COLOR["Implicit"],  label="Implicit"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=USER_COLOR,
                   markersize=7, label="user turn"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=ASSISTANT_COLOR,
                   markersize=7, label="assistant turn"),
    ]
    fig.legend(handles=legend_handles, loc="lower right",
               fontsize=8, framealpha=0.8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_combined(results: list[dict], outfile: str):
    """Single axes: all cases overlaid, coloured by condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for d in results:
        condition = d.get("condition", "Explicit")
        color = CONDITION_COLOR.get(condition, "#333333")
        steps = [s["step"] for s in d["per_step"]]

        ha = [s["harm_amplification"] for s in d["per_step"]]
        pp = [s["pref_prob"]          for s in d["per_step"]]

        axes[0].plot(steps, ha, color=color, linewidth=1.2, alpha=0.6)
        axes[1].plot(steps, pp, color=color, linewidth=1.2, alpha=0.6)

    axes[0].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    axes[0].set_title("Harm Amplification  Δ_s(y−) − Δ_s(y+)", fontsize=11)
    axes[0].set_xlabel("Step (odd=user, even=assistant)")
    axes[0].set_ylabel("Harm Amplification")

    axes[1].axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.5,
                    label="chance (0.5)")
    axes[1].set_title("Preference Probability  p(y+ ≻ y− | x, s)", fontsize=11)
    axes[1].set_xlabel("Step (odd=user, even=assistant)")
    axes[1].set_ylabel("p(y+ ≻ y−)")
    axes[1].set_ylim(0, 1)

    legend_handles = [
        mpatches.Patch(color=CONDITION_COLOR["Explicit"], label="Explicit"),
        mpatches.Patch(color=CONDITION_COLOR["Implicit"], label="Implicit"),
        mpatches.Patch(color="white", label=""),
        plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.5, label="baseline"),
    ]
    for ax in axes:
        ax.legend(handles=legend_handles, fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle("KL-Drift Across All Cases", fontsize=13, fontweight="bold")
    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


DELTA_COLORS = {
    "delta_ypos":    "#2ca02c",   # green  — safe response
    "delta_yneg":    "#d62728",   # red    — harmful response
    "delta_yrefuse": "#7f7f7f",   # grey   — terse refusal probe
}


def plot_three_way(results: list[dict], outfile: str):
    """Per-case subplot: all three delta trajectories on the same axes."""
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.4),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, d in enumerate(results):
        ax = axes[i]
        steps = [s["step"] for s in d["per_step"]]

        for key, color, label in [
            ("delta_ypos",    DELTA_COLORS["delta_ypos"],    "y+ (safe)"),
            ("delta_yneg",    DELTA_COLORS["delta_yneg"],    "y− (harmful)"),
            ("delta_yrefuse", DELTA_COLORS["delta_yrefuse"], "y_refusal"),
        ]:
            vals = [s[key] for s in d["per_step"]]
            ax.plot(steps, vals, color=color, linewidth=1.8,
                    label=label, zorder=3)

            # marker shape by role
            for s_val, v, role in zip(steps, vals, [s["role"] for s in d["per_step"]]):
                mk = "^" if role == "user" else ("s" if role == "assistant" else "o")
                ax.scatter(s_val, v, color=color, marker=mk, s=22, zorder=4)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

        # light vertical bands per turn
        for t in range(d["T"]):
            ax.axvspan(2 * t + 1, 2 * t + 2, alpha=0.04, color="gray")

        ax.set_title(short_title(d["name"]), fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Δ_s(y | x)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 2 * d["T"])
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "KL-Drift per response type  "
        "[▲=user step  ■=assistant step]\n"
        "green=y+ (safe)   red=y− (harmful)   grey=y_refusal (terse probe)",
        fontsize=11, fontweight="bold", y=1.02,
    )
    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_decomposition(results: list[dict], outfile: str):
    """
    Per-case subplot showing the two decomposition signals:
      harmful_compliance_drift  = Δ(y−) − Δ(y_refusal)   [above 0 = real compliance drift]
      safety_quality_erosion    = Δ(y_refusal) − Δ(y+)   [above 0 = safety quality eroding]
    """
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.4),
                             constrained_layout=True)
    axes = axes.flatten()

    for i, d in enumerate(results):
        ax = axes[i]
        steps = [s["step"] for s in d["per_step"]]
        roles = [s["role"]  for s in d["per_step"]]

        for key, color, label in [
            ("harmful_compliance_drift", "#d62728", "Δ(y−)−Δ(refusal)  compliance"),
            ("safety_quality_erosion",   "#ff7f0e", "Δ(refusal)−Δ(y+)  quality erosion"),
        ]:
            vals = [s[key] for s in d["per_step"]]
            ax.plot(steps, vals, color=color, linewidth=1.8, label=label, zorder=3)
            for s_val, v, role in zip(steps, vals, roles):
                mk = "^" if role == "user" else ("s" if role == "assistant" else "o")
                ax.scatter(s_val, v, color=color, marker=mk, s=22, zorder=4)

        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        for t in range(d["T"]):
            ax.axvspan(2 * t + 1, 2 * t + 2, alpha=0.04, color="gray")

        ax.set_title(short_title(d["name"]), fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Δ", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, 2 * d["T"])
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Drift decomposition  [▲=user  ■=assistant]\n"
        "red = harmful compliance beyond refusal   orange = safety quality erosion",
        fontsize=11, fontweight="bold", y=1.02,
    )
    out_path = OUT_DIR / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot KL-drift figures.")
    parser.add_argument("--results_dir", default="dp://harm_kl.results_kl_drift")
    parser.add_argument(
        "--data-paths-config",
        default="configs/data_paths.yaml",
        help="Path to data path registry yaml.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    data_paths = load_data_paths(args.data_paths_config)

    global RESULTS_DIR, OUT_DIR
    RESULTS_DIR = resolve_path_spec(args.results_dir, data_paths)
    OUT_DIR = RESULTS_DIR

    results = load_results()
    print(f"Loaded {len(results)} cases.")

    plot_metric(
        results,
        metric="harm_amplification",
        ylabel="Δ_s(y−) − Δ_s(y+)",
        title="Per-Step Harm Amplification  [red = harmful drift > safe drift]",
        outfile="harm_amplification.png",
        hline=0,
    )

    plot_metric(
        results,
        metric="pref_prob",
        ylabel="p(y+ ≻ y− | x, s)",
        title="Per-Step Preference Probability  [below 0.5 = harmful drift dominates]",
        outfile="pref_prob.png",
        hline=0.5,
        hline_label="chance",
    )

    plot_combined(results, outfile="overview.png")
    plot_three_way(results, outfile="three_way_delta.png")
    plot_decomposition(results, outfile="decomposition.png")


if __name__ == "__main__":
    main()
