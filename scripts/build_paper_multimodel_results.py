#!/usr/bin/env python3
"""Build paper-ready multi-model aggregates from the saved response analyses.

The response-generation run included three LMSYS targets that are not part of the
redistributable benchmark. This script joins every analysis row to generation
metadata and excludes those targets before computing any reported quantity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESPONSES = ROOT / "results/response_analysis/10model_low_reasoning/latest_success_responses.jsonl"
ANNOTATIONS = ROOT / (
    "results/generated_response_annotations/gpt-5.4-mini/combined_8_flags/"
    "response_annotation_matrix.csv"
)
TOPIC_ASSIGNMENTS = ROOT / (
    "results/response_analysis/10model_low_reasoning/fine_topics_40/"
    "fine_topic_assignments.csv"
)
TOPIC_TABLE = ROOT / (
    "results/response_analysis/10model_low_reasoning/fine_topics_40/"
    "simple_lda_report/simple_lda_topic_table.csv"
)
OUTPUT = ROOT / "paper/iclr2026/artifacts/multimodel"
FIGURES = ROOT / "paper/iclr2026/figures"

MODEL_ORDER = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    "gpt-4.1-mini-2025-04-14",
    "gpt-5-mini-2025-08-07",
    "gpt-5.2-2025-12-11",
    "gpt-5.5-2026-04-23",
]

MODEL_LABELS = {
    "gpt-3.5-turbo-0125": "GPT-3.5\nJan 2024",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo\nApr 2024",
    "gpt-4o-2024-05-13": "GPT-4o\nMay 2024",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini\nJul 2024",
    "o1-2024-12-17": "o1\nDec 2024",
    "o3-mini-2025-01-31": "o3 mini\nJan 2025",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 mini\nApr 2025",
    "gpt-5-mini-2025-08-07": "GPT-5 mini\nAug 2025",
    "gpt-5.2-2025-12-11": "GPT-5.2\nDec 2025",
    "gpt-5.5-2026-04-23": "GPT-5.5\nApr 2026",
}

FLAG_LABELS = {
    "bot_endorses_delusion": "Endorses claim",
    "bot_grand_significance": "Grand significance",
    "bot_dismisses_counterevidence": "Dismisses counterevidence",
    "bot_metaphysical_themes": "Metaphysical themes",
    "bot_reflective_summary": "Reflective summary",
    "bot_positive_affirmation": "Positive affirmation",
    "bot_misrepresents_sentience": "Misrepresents sentience",
    "bot_claims_unique_connection": "Claims unique connection",
}


def wilson_interval(positive: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return math.nan, math.nan
    p = positive / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half_width, center + half_width


def load_generation_metadata() -> pd.DataFrame:
    rows = []
    with RESPONSES.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "generation_id": row["generation_id"],
                    "model": row["model"],
                    "source": row["source"],
                    "conversation_id": row["conversation_id"],
                    "message_hash": row["message_hash"],
                }
            )
    metadata = pd.DataFrame(rows)
    if metadata["generation_id"].duplicated().any():
        raise ValueError("Generation IDs must be unique")
    return metadata


def aggregate_binary_flags(frame: pd.DataFrame, flag_columns: list[str]) -> pd.DataFrame:
    rows = []
    for model in [None, *MODEL_ORDER]:
        subset = frame if model is None else frame.loc[frame["model"] == model]
        for column in flag_columns:
            positive = int(subset[column].astype(bool).sum())
            total = len(subset)
            lower, upper = wilson_interval(positive, total)
            rows.append(
                {
                    "model": "all" if model is None else model,
                    "flag": column.removesuffix("_positive"),
                    "label": FLAG_LABELS[column.removesuffix("_positive")],
                    "positive": positive,
                    "total": total,
                    "rate": positive / total,
                    "wilson_lower": lower,
                    "wilson_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def aggregate_topics(assignments: pd.DataFrame, topic_labels: dict[int, str]) -> pd.DataFrame:
    rows = []
    for model in [None, *MODEL_ORDER]:
        subset = assignments if model is None else assignments.loc[assignments["model"] == model]
        counts = subset["topic"].value_counts()
        for topic, label in topic_labels.items():
            positive = int(counts.get(topic, 0))
            total = len(subset)
            lower, upper = wilson_interval(positive, total)
            rows.append(
                {
                    "model": "all" if model is None else model,
                    "topic": topic,
                    "label": label,
                    "positive": positive,
                    "total": total,
                    "rate": positive / total,
                    "wilson_lower": lower,
                    "wilson_upper": upper,
                }
            )
    return pd.DataFrame(rows)


def plot_taxonomy(aggregates: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (overall_axis, model_axis) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    overall = aggregates.loc[aggregates["model"] == "all"].sort_values("rate")
    y = np.arange(len(overall))
    overall_axis.barh(y, 100 * overall["rate"], color="#3274A1", alpha=0.9)
    overall_axis.errorbar(
        100 * overall["rate"],
        y,
        xerr=np.vstack(
            [
                100 * (overall["rate"] - overall["wilson_lower"]),
                100 * (overall["wilson_upper"] - overall["rate"]),
            ]
        ),
        fmt="none",
        color="#222222",
        capsize=3,
        linewidth=1,
    )
    overall_axis.set_yticks(y, overall["label"])
    overall_axis.set_xlabel("Share of assistant responses (%)")
    overall_axis.set_title("A  Overall SPIRALS taxonomy prevalence", loc="left", fontweight="bold")

    selected = [
        "bot_endorses_delusion",
        "bot_grand_significance",
        "bot_dismisses_counterevidence",
        "bot_misrepresents_sentience",
    ]
    colors = ["#C44E52", "#DD8452", "#4C72B0", "#55A868"]
    x = np.arange(len(MODEL_ORDER))
    for flag, color in zip(selected, colors, strict=True):
        data = aggregates.loc[
            (aggregates["model"] != "all") & (aggregates["flag"] == flag)
        ].set_index("model").loc[MODEL_ORDER]
        rate = 100 * data["rate"].to_numpy()
        errors = np.vstack(
            [
                100 * (data["rate"] - data["wilson_lower"]).to_numpy(),
                100 * (data["wilson_upper"] - data["rate"]).to_numpy(),
            ]
        )
        model_axis.errorbar(
            x,
            rate,
            yerr=errors,
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=2,
            color=color,
            label=FLAG_LABELS[flag],
        )
    model_axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=45, ha="right")
    model_axis.set_ylabel("Share of assistant responses (%)")
    model_axis.set_title("B  Selected flags across model snapshots", loc="left", fontweight="bold")
    model_axis.legend(frameon=False, fontsize=8, loc="upper left")

    figure.suptitle("SPIRALS response taxonomy on 4,316 public-cohort generations", fontweight="bold")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"multimodel_spirals_taxonomy.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_topics(aggregates: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (overall_axis, model_axis) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    overall = (
        aggregates.loc[aggregates["model"] == "all"].nlargest(10, "rate").sort_values("rate")
    )
    y = np.arange(len(overall))
    overall_axis.barh(y, 100 * overall["rate"], color="#4C956C", alpha=0.9)
    overall_axis.errorbar(
        100 * overall["rate"],
        y,
        xerr=np.vstack(
            [
                100 * (overall["rate"] - overall["wilson_lower"]),
                100 * (overall["wilson_upper"] - overall["rate"]),
            ]
        ),
        fmt="none",
        color="#222222",
        capsize=3,
        linewidth=1,
    )
    overall_axis.set_yticks(y, overall["label"])
    overall_axis.set_xlabel("Assigned responses (%)")
    overall_axis.set_title("A  Ten most prevalent LDA topics", loc="left", fontweight="bold")

    selected_topics = overall.nlargest(5, "rate")["topic"].tolist()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    x = np.arange(len(MODEL_ORDER))
    for topic, color in zip(selected_topics, colors, strict=True):
        data = aggregates.loc[
            (aggregates["model"] != "all") & (aggregates["topic"] == topic)
        ].set_index("model").loc[MODEL_ORDER]
        rate = 100 * data["rate"].to_numpy()
        errors = np.vstack(
            [
                100 * (data["rate"] - data["wilson_lower"]).to_numpy(),
                100 * (data["wilson_upper"] - data["rate"]).to_numpy(),
            ]
        )
        model_axis.errorbar(
            x,
            rate,
            yerr=errors,
            marker="o",
            markersize=4,
            linewidth=1.6,
            capsize=2,
            color=color,
            label=data["label"].iloc[0],
        )
    model_axis.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=45, ha="right")
    model_axis.set_ylabel("Assigned responses (%)")
    model_axis.set_title("B  Five leading topics across model snapshots", loc="left", fontweight="bold")
    model_axis.legend(frameon=False, fontsize=8, loc="upper right")

    figure.suptitle("Descriptive 40-topic LDA of assistant responses", fontweight="bold")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"multimodel_lda_topics.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    metadata = load_generation_metadata()
    public = metadata.loc[metadata["source"] != "lmsys_chat_1m"].copy()
    if len(public) != 4316 or public["message_hash"].nunique() != 433:
        raise ValueError(
            f"Expected 4,316 generations over 433 public targets; got "
            f"{len(public):,} over {public['message_hash'].nunique():,}"
        )

    annotations = pd.read_csv(ANNOTATIONS).merge(
        public[["generation_id", "model", "source", "conversation_id", "message_hash"]],
        on="generation_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_metadata"),
    )
    if len(annotations) != len(public):
        raise ValueError("Not every public generation has a SPIRALS annotation row")
    if not (annotations["generated_model"] == annotations["model"]).all():
        raise ValueError("Annotation and generation model IDs disagree")

    flag_columns = [column for column in annotations if column.endswith("_positive")]
    flag_aggregates = aggregate_binary_flags(annotations, flag_columns)
    flag_aggregates.to_csv(OUTPUT / "spirals_taxonomy_by_model.csv", index=False)

    assignments = pd.read_csv(TOPIC_ASSIGNMENTS).merge(
        public[["generation_id", "source", "conversation_id", "message_hash"]],
        on="generation_id",
        how="inner",
        validate="one_to_one",
    )
    if len(assignments) != len(public):
        raise ValueError("Not every public generation has an LDA assignment")
    topic_table = pd.read_csv(TOPIC_TABLE)
    topic_labels = dict(zip(topic_table["topic_id"].astype(int), topic_table["category"], strict=True))
    topic_aggregates = aggregate_topics(assignments, topic_labels)
    topic_aggregates.to_csv(OUTPUT / "lda_topics_by_model.csv", index=False)

    plot_taxonomy(flag_aggregates)
    plot_topics(topic_aggregates)

    summary = {
        "analysis_population": {
            "responses": len(public),
            "targets": int(public["message_hash"].nunique()),
            "conversations": int(public["conversation_id"].nunique()),
            "models": len(MODEL_ORDER),
            "excluded_lmsys_responses": int(len(metadata) - len(public)),
        },
        "models": {
            model: int((public["model"] == model).sum()) for model in MODEL_ORDER
        },
        "taxonomy_overall": flag_aggregates.loc[flag_aggregates["model"] == "all"]
        .sort_values("rate", ascending=False)
        .to_dict("records"),
        "lda_top_topics": topic_aggregates.loc[topic_aggregates["model"] == "all"]
        .nlargest(10, "rate")
        .to_dict("records"),
        "notes": [
            "All rows are successful generations for the 433 public primary-cohort targets.",
            "The three excluded LMSYS targets account for 30 generations.",
            "SPIRALS flags use GPT-5.4-mini and the pinned package prompts.",
            "LDA uses the fixed 40-topic fit; only post-fit reported aggregates are source-filtered.",
            "Intervals are pointwise 95% Wilson score intervals and are not multiple-comparison adjusted.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
