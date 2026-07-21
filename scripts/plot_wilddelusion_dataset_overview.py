from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_pipeline(ax: plt.Axes, settings: dict[str, Any], summary: dict[str, Any]) -> None:
    stages = [
        ("6.13M", "embedded\nuser messages"),
        ("1,000", "bootstrap\njudgments"),
        ("107", "seed true\npositives"),
        ("3,000", "whitened\nretrieval"),
        (f"{summary['input_context_rows']}", "candidate target\ncontexts"),
        (f"{summary['verified_positive_rows']}", "verified target\nturns"),
        (f"{summary['released_rows']}", "redistributable\ntarget turns"),
    ]
    expected = int(round(settings["materialized_embedding_corpus_rows_approx"] / 10_000))
    if expected != 613:
        raise ValueError("The figure label assumes approximately 6.13M corpus rows")

    x_positions = np.linspace(0.08, 0.92, len(stages))
    for index, (x, (count, label)) in enumerate(zip(x_positions, stages, strict=True)):
        color = "#2A6F97" if index in {0, len(stages) - 1} else "#6C757D"
        box = FancyBboxPatch(
            (x - 0.055, 0.35),
            0.11,
            0.34,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x, 0.57, count, ha="center", va="center", fontsize=15, weight="bold", color=color)
        ax.text(x, 0.42, label, ha="center", va="center", fontsize=9.5, linespacing=1.05)
        if index < len(stages) - 1:
            next_x = x_positions[index + 1]
            ax.annotate(
                "",
                xy=(next_x - 0.064, 0.52),
                xytext=(x + 0.064, 0.52),
                arrowprops={"arrowstyle": "->", "color": "#8B8B8B", "lw": 1.4},
            )
    ax.text(
        0.5,
        0.13,
        "Embedding retrieval enriches a rare class; conversation context removes role-play, fiction, and text tasks.",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def horizontal_bars(
    ax: plt.Axes,
    labels: list[str],
    values: list[int],
    color: str,
    denominator: int | None = None,
) -> None:
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=color, height=0.62)
    ax.set_yticks(y, labels)
    maximum = max(values)
    ax.set_xlim(0, maximum * 1.28)
    for position, value in zip(y, values, strict=True):
        suffix = f" ({100 * value / denominator:.1f}%)" if denominator else ""
        ax.text(value + maximum * 0.025, position, f"{value}{suffix}", va="center", fontsize=10.5)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the WildDelusion mining and verification overview.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("results/positive_bootstrap_v1/settings.json"),
    )
    parser.add_argument(
        "--verification-summary",
        type=Path,
        default=Path("results/whitened_top3k_verification/summary.json"),
    )
    parser.add_argument(
        "--release-characterization",
        type=Path,
        default=Path("paper/iclr2026/artifacts/release_characterization.json"),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("data/releases/WildDelusionVerified/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/whitened_top3k_verification/dataset_overview.png"),
    )
    args = parser.parse_args()

    settings = read_json(args.settings)
    summary = read_json(args.verification_summary)
    release = read_json(args.release_manifest)
    characterization = read_json(args.release_characterization)
    summary["released_rows"] = release["rows"]
    source_counts = release["source_target_turn_counts"]
    sources = ["ShareChat-ChatGPT", "WildChat", "ShareChat-Grok"]
    source_values = [
        source_counts["sharechat_chatgpt"],
        source_counts["wildchat_full"],
        source_counts["sharechat_grok"],
    ]
    exclusions = summary["exclusion_counts"]
    exclusion_order = [
        ("Role-play", "roleplay"),
        ("Ordinary plausible", "ordinary_plausible"),
        ("Fiction/story", "fiction_or_story"),
        ("Joke/absurd", "joke_or_absurd"),
        ("Translation/text task", "translation_or_text_task"),
        ("Insufficient context", "insufficient_context"),
        ("Third party/quoted", "third_party_or_quoted"),
        ("Dream report", "dreams"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10.5,
        }
    )
    fig = plt.figure(figsize=(16.8, 7.0))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.0, 1.08, 0.92],
        hspace=0.22,
        wspace=0.58,
    )
    pipeline_ax = fig.add_subplot(grid[0, :])
    source_ax = fig.add_subplot(grid[1, 0])
    exclusion_ax = fig.add_subplot(grid[1, 1])
    concentration_ax = fig.add_subplot(grid[1, 2])

    draw_pipeline(pipeline_ax, settings, summary)
    pipeline_ax.set_title("A  Mining a rare behavior from three open conversation corpora", loc="left", weight="bold")

    horizontal_bars(
        source_ax,
        sources,
        source_values,
        "#2A6F97",
        denominator=release["rows"],
    )
    source_ax.set_title("B  Context-verified target turns by source", loc="left", weight="bold")
    source_ax.set_xlabel("Retained target turns")

    horizontal_bars(
        exclusion_ax,
        [label for label, _ in exclusion_order],
        [exclusions[key] for _, key in exclusion_order],
        "#C96A3D",
    )
    exclusion_ax.set_title("C  Contextual exclusions", loc="left", weight="bold")
    exclusion_ax.set_xlabel("Excluded or uncertain conversations")

    conversation_n = characterization["distinct_source_conversations"]
    multi_target_n = characterization["multi_target_source_conversations"]
    single_target_n = conversation_n - multi_target_n
    horizontal_bars(
        concentration_ax,
        ["One retained target", "Multiple retained targets"],
        [single_target_n, multi_target_n],
        "#3A7D44",
        denominator=conversation_n,
    )
    concentration_ax.set_xlabel("Source conversations")
    concentration_ax.set_title("D  Conversation concentration", loc="left", weight="bold")
    concentration_ax.text(
        0,
        -0.72,
        "Maximum: 39 retained targets in one conversation",
        fontsize=9.5,
        color="#555555",
    )

    fig.suptitle("WildDelusion construction and composition", fontsize=19, weight="bold", y=0.995)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.1, top=0.92)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
