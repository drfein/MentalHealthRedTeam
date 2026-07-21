from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_LABELS = {
    "direct_assertion": "Direct assertion",
    "reported_belief": "Reported belief",
    "reconsideration": "Reconsideration",
    "quotation_analysis": "Quotation analysis",
    "question": "Question",
    "explicit_fiction": "Explicit fiction",
    "skeptical_roleplay": "Skeptical role-play",
    "translation": "Translation",
}

ENDPOINT_LABELS = {
    "framing_rubric": "Framing-aware rubric",
    "exact_package_rubric": "Exact SPIRALS rubric",
}


def horizontal_interval(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    estimate: str,
    low: str,
    high: str,
    colors: list[str] | str,
) -> None:
    values = frame[estimate].to_numpy(float)
    color_values = [colors] * len(frame) if isinstance(colors, str) else colors
    for index, (value, low_value, high_value, color) in enumerate(
        zip(values, frame[low], frame[high], color_values, strict=True)
    ):
        ax.errorbar(
            value,
            index,
            xerr=[[value - low_value], [high_value - value]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            linewidth=1.6,
            markersize=6,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot corrected behavioral and J-space paper results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/jspace_semantic_specificity/qwen2_5_7b_counterfactuals"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    correction = args.results_root / "conversation_disjoint_correction"
    behavior = pd.read_csv(correction / "behavior_by_frame.csv")
    behavior = behavior[behavior["population"].eq("original_holdout")].sort_values(
        "positive_rate", ascending=False
    )
    endpoints = pd.read_csv(correction / "endpoint_performance.csv")
    endpoints = endpoints[
        endpoints["population"].eq("conversation_disjoint_holdout")
    ].copy()
    endpoint_order = ["framing_rubric", "exact_package_rubric"]
    endpoints = endpoints.set_index("endpoint").loc[endpoint_order].reset_index()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titleweight": "bold",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), gridspec_kw={"wspace": 0.52})

    behavior_colors = [
        "#245c8a" if arm == "direct_assertion" else "#777777"
        for arm in behavior["arm"]
    ]
    horizontal_interval(
        axes[0],
        behavior,
        estimate="positive_rate",
        low="ci_low",
        high="ci_high",
        colors=behavior_colors,
    )
    axes[0].set_yticks(
        np.arange(len(behavior)), [ARM_LABELS[arm] for arm in behavior["arm"]]
    )
    axes[0].set_xlabel("Assistant endorsement rate")
    axes[0].set_title("A  Matched framing changes behavior", loc="left")
    axes[0].set_xlim(-0.005, 0.27)

    horizontal_interval(
        axes[1],
        endpoints,
        estimate="auc",
        low="ci_low",
        high="ci_high",
        colors="#7a5195",
    )
    axes[1].axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_yticks(
        np.arange(len(endpoints)),
        [
            f"{ENDPOINT_LABELS[row.endpoint]} (n+={row.positive_n})"
            for row in endpoints.itertuples(index=False)
        ],
    )
    axes[1].set_xlabel("AUROC of frozen layer-26 readout")
    axes[1].set_title("B  Association on unseen conversations", loc="left")
    axes[1].set_xlim(0.43, 1.0)
    axes[1].text(
        0.98,
        0.07,
        (
            f"{int(endpoints.iloc[0]['n_target_turns'])} target turns from "
            f"{int(endpoints.iloc[0]['n_source_conversations'])} conversations"
        ),
        transform=axes[1].transAxes,
        ha="right",
        color="#444444",
        fontsize=11,
    )

    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Controlled behavior and conversation-disjoint J-space sensitivity",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.86)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
