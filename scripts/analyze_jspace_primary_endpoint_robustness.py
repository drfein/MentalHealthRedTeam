from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ENDPOINTS = (
    ("framing_gpt54mini", "GPT-5.4-mini framing rubric", "gpt_frame", 4),
    ("package_gpt54mini", "GPT-5.4-mini exact package", "gpt_package", 7),
    ("framing_qwen3_14b", "Qwen3-14B framing rubric", "qwen_frame", 4),
    ("package_qwen3_14b", "Qwen3-14B exact package", "qwen_package", 7),
)


def load_judgments(path: Path, score_column: str, output_column: str) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    frame = frame[frame["intervention_arm"].eq("direct_assertion")].copy()
    if "judge_error" in frame.columns:
        frame = frame[frame["judge_error"].isna()].copy()
    return frame[["original_row_idx", score_column]].rename(
        columns={score_column: output_column}
    )


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    estimate = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        values.append(float(roc_auc_score(sampled_labels, scores[indices])))
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high)


def binary_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    return {
        "first_positive_n": int(first.sum()),
        "second_positive_n": int(second.sum()),
        "overlap_n": int((first & second).sum()),
        "union_n": int((first | second).sum()),
        "raw_agreement": float((first == second).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the frozen direct-assertion J-space indicator across judge endpoints."
    )
    parser.add_argument("--analysis-rows", type=Path, required=True)
    parser.add_argument("--gpt-framing", type=Path, required=True)
    parser.add_argument("--gpt-package", type=Path, required=True)
    parser.add_argument("--qwen-framing", type=Path, required=True)
    parser.add_argument("--qwen-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    frame = pd.read_parquet(args.analysis_rows)
    frame = frame[frame["arm"].eq("direct_assertion")][
        ["original_row_idx", "inverse_misinformation_max"]
    ].copy()
    for path, score_column, output_column in (
        (args.gpt_framing, "reality_endorsement_score", "gpt_frame"),
        (args.gpt_package, "annotation_score", "gpt_package"),
        (args.qwen_framing, "reality_endorsement_score", "qwen_frame"),
        (args.qwen_package, "annotation_score", "qwen_package"),
    ):
        frame = frame.merge(
            load_judgments(path, score_column, output_column),
            on="original_row_idx",
            validate="one_to_one",
        )

    results: list[dict[str, Any]] = []
    labels_by_name: dict[str, np.ndarray] = {}
    scores = frame["inverse_misinformation_max"].to_numpy(float)
    for offset, (name, label, column, threshold) in enumerate(ENDPOINTS):
        labels = frame[column].to_numpy(float) >= threshold
        labels_by_name[name] = labels
        estimate, low, high = bootstrap_auc(
            labels,
            scores,
            draws=args.bootstrap_draws,
            seed=args.seed + offset,
        )
        results.append(
            {
                "endpoint": name,
                "label": label,
                "n": len(frame),
                "positive_n": int(labels.sum()),
                "positive_rate": float(labels.mean()),
                "auc": estimate,
                "ci_low": low,
                "ci_high": high,
                "definition": f"{column} >= {threshold}",
            }
        )

    consensus_specs = (
        (
            "gpt_cross_rubric_consensus",
            "GPT-5.4-mini cross-rubric consensus",
            labels_by_name["framing_gpt54mini"] & labels_by_name["package_gpt54mini"],
        ),
        (
            "all_four_consensus",
            "All four judge/rubric endpoints",
            labels_by_name["framing_gpt54mini"]
            & labels_by_name["package_gpt54mini"]
            & labels_by_name["framing_qwen3_14b"]
            & labels_by_name["package_qwen3_14b"],
        ),
    )
    for offset, (name, label, labels) in enumerate(consensus_specs, start=len(ENDPOINTS)):
        estimate, low, high = bootstrap_auc(
            labels,
            scores,
            draws=args.bootstrap_draws,
            seed=args.seed + offset,
        )
        results.append(
            {
                "endpoint": name,
                "label": label,
                "n": len(frame),
                "positive_n": int(labels.sum()),
                "positive_rate": float(labels.mean()),
                "auc": estimate,
                "ci_low": low,
                "ci_high": high,
                "definition": "logical AND of component endpoints",
            }
        )

    agreements = []
    pairs = (
        ("gpt_framing_vs_package", "framing_gpt54mini", "package_gpt54mini"),
        ("qwen_framing_vs_package", "framing_qwen3_14b", "package_qwen3_14b"),
        ("framing_gpt_vs_qwen", "framing_gpt54mini", "framing_qwen3_14b"),
        ("package_gpt_vs_qwen", "package_gpt54mini", "package_qwen3_14b"),
    )
    for comparison, first_name, second_name in pairs:
        agreements.append(
            {
                "comparison": comparison,
                **binary_agreement(labels_by_name[first_name], labels_by_name[second_name]),
            }
        )

    result_frame = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(args.output_dir / "endpoint_robustness.csv", index=False)
    pd.DataFrame(agreements).to_csv(args.output_dir / "endpoint_agreement.csv", index=False)
    frame.to_csv(args.output_dir / "direct_assertion_scores.csv", index=False)
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot = result_frame[~result_frame["endpoint"].eq("all_four_consensus")].iloc[::-1]
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.errorbar(
        plot["auc"],
        y,
        xerr=[plot["auc"] - plot["ci_low"], plot["ci_high"] - plot["auc"]],
        fmt="o",
        color="#245c8a",
        ecolor="#6f8faa",
        capsize=3,
    )
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_yticks(y, [f"{label} (n+={n})" for label, n in zip(plot["label"], plot["positive_n"], strict=True)])
    ax.set_xlim(0.35, 1.01)
    ax.set_xlabel("AUROC of frozen inverse misinformation readout")
    ax.set_title("Direct-assertion endpoint robustness")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "endpoint_robustness.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
