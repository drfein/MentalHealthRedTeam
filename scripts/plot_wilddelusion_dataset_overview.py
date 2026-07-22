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


def draw_pipeline(
    ax: plt.Axes,
    settings: dict[str, Any],
    summary: dict[str, Any],
    combined: dict[str, Any],
) -> None:
    split_counts = combined["discovery_split_target_counts"]
    stages = [
        ("6.13M", "embedded\nuser messages"),
        ("1,000", "bootstrap\njudgments"),
        ("107", "seed judged\npositives"),
        ("3,000", "whitened\nretrieval"),
        (f"{split_counts['openai_embedding_whitened']}", "primary released\ntarget turns"),
        (f"+{split_counts['legacy_probe_gpt52']}", "non-overlapping\nlegacy probe turns"),
        (f"{combined['rows']}", "combined released\ntarget turns"),
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
        "Two retrieval routes are kept explicit; both use the same current message and context verification rules.",
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
        default=Path("paper/iclr2026/artifacts/combined_release_characterization.json"),
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("paper/iclr2026/artifacts/combined_release_manifest.json"),
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
    source_counts = release["source_target_counts"]
    sources = [
        "ShareChat-ChatGPT",
        "WildChat",
        "ShareChat-Grok",
        "ShareChat-Gemini",
        "ShareChat-Claude",
    ]
    source_values = [
        source_counts["sharechat_chatgpt"],
        source_counts["wildchat_full"],
        source_counts["sharechat_grok"],
        source_counts["sharechat_gemini"],
        source_counts["sharechat_claude"],
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
    split_ax = fig.add_subplot(grid[1, 1])
    concentration_ax = fig.add_subplot(grid[1, 2])

    draw_pipeline(pipeline_ax, settings, summary, release)
    pipeline_ax.set_title(
        "A  Mining a rare behavior with two retrieval routes", loc="left", weight="bold"
    )

    horizontal_bars(
        source_ax,
        sources,
        source_values,
        "#2A6F97",
        denominator=release["rows"],
    )
    source_ax.set_title("B  Context-verified target turns by source", loc="left", weight="bold")
    source_ax.set_xlabel("Retained target turns")

    split_labels = ["Whitened embedding", "Legacy probe"]
    split_values = [
        release["discovery_split_target_counts"]["openai_embedding_whitened"],
        release["discovery_split_target_counts"]["legacy_probe_gpt52"],
    ]
    horizontal_bars(split_ax, split_labels, split_values, "#C96A3D", denominator=release["rows"])
    split_ax.set_title("C  Explicit discovery splits", loc="left", weight="bold")
    split_ax.set_xlabel("Retained target turns")

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
        f"Maximum: {int(characterization['target_turns_per_source_conversation']['max'])} retained targets in one conversation",
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
