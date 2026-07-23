from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wild_delusion_miner.plot_style import (  # noqa: E402
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    apply_paper_style,
)


def plot_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    omnibus: pd.DataFrame,
    split: str,
) -> None:
    split_frame = frame[frame["discovery_split"] == split].set_index("model")
    model_order = [model for model in MODEL_ORDER if model in split_frame.index]
    subset = split_frame.reindex(model_order)
    summary = omnibus[omnibus["discovery_split"] == split].iloc[0]
    ax.plot(
        [summary["ci_low_pp"], summary["ci_high_pp"]],
        [0, 0],
        color="#111827",
        linewidth=2.4,
        solid_capstyle="round",
    )
    ax.scatter(
        summary["difference_pp"],
        0,
        marker="D",
        s=66,
        color="#111827",
        zorder=4,
    )
    ax.axhline(0.6, color="#D1D5DB", linewidth=0.9)

    for index, (model, row) in enumerate(subset.iterrows()):
        position = index + 1
        color = MODEL_COLORS[model]
        ax.plot(
            [row["ci_low_pp"], row["ci_high_pp"]],
            [position, position],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            alpha=0.9,
        )
        ax.scatter(
            row["difference_pp"],
            position,
            s=58,
            facecolor=color,
            edgecolor=color,
            linewidth=1.6,
            zorder=3,
        )
    ax.axvline(0, color="#6B7280", linewidth=1.0)
    ax.set_yticks(
        range(len(subset) + 1),
        ["Overall", *[MODEL_LABELS[model] for model in subset.index]],
    )
    ax.invert_yaxis()
    ax.set_xlim(-5, 25)
    ax.set_xticks([-5, 0, 5, 10, 15, 20, 25])
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Increase in delusion endorsement rate (percent)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot model-level combined context endpoint effects."
    )
    parser.add_argument(
        "--inference",
        type=Path,
        default=Path(
            "results/combined_context_endpoints/analysis/endpoint_inference.csv"
        ),
    )
    parser.add_argument(
        "--omnibus",
        type=Path,
        default=Path(
            "results/combined_context_endpoints/analysis/omnibus_inference.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/combined_context_endpoints/analysis/context_endpoint_forest.png"
        ),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.inference)
    omnibus = pd.read_csv(args.omnibus)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    plot_panel(ax, frame, omnibus, "combined")
    fig.subplots_adjust(top=0.98, bottom=0.13, left=0.23, right=0.96)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
