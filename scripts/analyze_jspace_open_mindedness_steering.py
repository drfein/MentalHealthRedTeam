#!/usr/bin/env python3
"""Analyze the paired open-mindedness-axis steering experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ["primary_down", "baseline", "primary_up", "random_down", "random_up"]
DISPLAY = {
    "primary_down": "Sensitivity -",
    "baseline": "Baseline",
    "primary_up": "Sensitivity +",
    "random_down": "Random -",
    "random_up": "Random +",
}
CORE_METRICS = [
    "reality_endorsement_score",
    "epistemic_sensitivity_score",
    "epistemic_openness_score",
    "respectful_engagement_score",
    "blanket_refusal",
    "incoherent",
]


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return pd.DataFrame(json.loads(line) for line in handle if line.strip())


def bootstrap_means(values: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    estimates = counts @ values / counts.sum(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def paired(frame: pd.DataFrame, metric: str, left: str, right: str) -> np.ndarray:
    wide = frame.pivot(index="original_row_idx", columns="condition", values=metric)
    return (wide[left].astype(float) - wide[right].astype(float)).to_numpy()


def merge_package_judgments(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    package = read_jsonl(path)
    keys = ["original_row_idx", "condition"]
    if package.duplicated(keys).any():
        raise ValueError("Package judgments contain duplicate item-condition rows")
    package = package[keys + ["annotation_score"]].rename(
        columns={"annotation_score": "package_endorsement_score"}
    )
    merged = frame.merge(package, on=keys, how="left", validate="one_to_one")
    if merged["package_endorsement_score"].isna().any():
        raise ValueError("Package judgments do not cover every custom-judge row")
    return merged


def summarize(
    frame: pd.DataFrame,
    draws: int,
    seed: int,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    if set(frame["condition"]) != set(CONDITIONS):
        raise ValueError(f"Unexpected conditions: {sorted(frame['condition'].unique())}")
    n = frame["original_row_idx"].nunique()
    if len(frame) != n * len(CONDITIONS):
        raise ValueError("Expected one complete five-condition set per message")
    metrics = metrics or CORE_METRICS
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=draws)
    summary_rows: list[dict[str, Any]] = []
    contrasts: dict[str, Any] = {}
    for condition in CONDITIONS:
        part = frame[frame["condition"].eq(condition)].sort_values("original_row_idx")
        row: dict[str, Any] = {"condition": condition, "n": len(part)}
        for metric in metrics:
            values = part[metric].astype(float).to_numpy()
            mean, low, high = bootstrap_means(values, counts)
            row.update({f"{metric}_mean": mean, f"{metric}_low": low, f"{metric}_high": high})
        summary_rows.append(row)
    for metric in metrics:
        target = paired(frame, metric, "primary_up", "primary_down")
        random = paired(frame, metric, "random_up", "random_down")
        for name, values in (
            ("axis_up_minus_down", target),
            ("random_up_minus_down", random),
            ("difference_in_differences", target - random),
        ):
            mean, low, high = bootstrap_means(values, counts)
            contrasts[f"{metric}:{name}"] = {
                "estimate": mean,
                "ci_low": low,
                "ci_high": high,
            }
    return {
        "n_messages": n,
        "n_observations": len(frame),
        "bootstrap_draws": draws,
        "bootstrap_unit": "held-out user message",
        "seed": seed,
        "metrics": metrics,
        "condition_summary": summary_rows,
        "contrasts": contrasts,
    }


def plot(summary: dict[str, Any], output: Path) -> None:
    frame = pd.DataFrame(summary["condition_summary"]).set_index("condition").loc[CONDITIONS]
    panels = [
        ("epistemic_sensitivity_score", "Epistemic sensitivity", (0, 5)),
        ("epistemic_openness_score", "Epistemic openness", (0, 5)),
        ("reality_endorsement_score", "Reality endorsement", (0, 5)),
        ("respectful_engagement_score", "Respectful engagement", (0, 5)),
        ("blanket_refusal", "Blanket refusal rate", (0, None)),
        ("incoherent", "Incoherence rate", (0, None)),
    ]
    colors = ["#2C7BB6", "#555555", "#D7191C", "#8C8C8C", "#8C8C8C"]
    x = np.arange(len(CONDITIONS))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    for ax, (metric, title, ylim) in zip(axes.flat, panels, strict=True):
        means = frame[f"{metric}_mean"].to_numpy()
        lows = frame[f"{metric}_low"].to_numpy()
        highs = frame[f"{metric}_high"].to_numpy()
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            color="#444444",
            capsize=3,
        )
        ax.scatter(x, means, c=colors, s=45, zorder=3)
        ax.set_xticks(x, [DISPLAY[item] for item in CONDITIONS], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Llama-3.1-8B J-space sensitivity versus openness steering", weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--package-judgments",
        type=Path,
        default=None,
        help="Optional exact bot-endorses-delusion package judgments to merge.",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    frame = read_jsonl(args.input)
    if "judge_error" in frame:
        frame = frame[frame["judge_error"].isna()].copy()
    metrics = list(CORE_METRICS)
    if args.package_judgments is not None:
        frame = merge_package_judgments(frame, args.package_judgments)
        metrics.append("package_endorsement_score")
    summary = summarize(frame, args.bootstrap_draws, args.seed, metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["condition_summary"]).to_csv(
        args.output_dir / "condition_summary.csv", index=False
    )
    (args.output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot(summary, args.output_dir / "open_mindedness_axis_steering.png")


if __name__ == "__main__":
    main()
