from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return float(center - radius), float(center + radius)


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size == 2:
            estimates.append(roc_auc_score(sampled_labels, scores[indices]))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def metric_row(name: str, numerator: int, denominator: int) -> dict[str, Any]:
    low, high = wilson_interval(numerator, denominator)
    return {
        "metric": name,
        "estimate": numerator / denominator,
        "ci_low": low,
        "ci_high": high,
        "numerator": numerator,
        "denominator": denominator,
        "interval": "95% Wilson",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the conversation verifier against an earlier human review set."
    )
    parser.add_argument("--human-reviews", type=Path, required=True)
    parser.add_argument("--candidate-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    reviews = pd.read_csv(args.human_reviews)
    candidates = pd.read_parquet(args.candidate_metadata)
    required_review = {"conversation_id", "flagged_text", "decision"}
    required_candidate = {
        "conversation_id",
        "flagged_text",
        "judge_label",
        "judge_confidence",
        "judge_keep_score",
    }
    if missing := required_review - set(reviews):
        raise ValueError(f"Human review file is missing columns: {sorted(missing)}")
    if missing := required_candidate - set(candidates):
        raise ValueError(f"Candidate metadata is missing columns: {sorted(missing)}")

    for frame in (reviews, candidates):
        frame["_conversation_id"] = frame["conversation_id"].astype(str)
        frame["_normalized_text"] = frame["flagged_text"].map(normalize_text)
    keys = ["_conversation_id", "_normalized_text"]
    candidate_columns = keys + [
        "judge_label",
        "judge_confidence",
        "judge_keep_score",
        "judge_exclusion",
    ]
    matched = reviews.merge(
        candidates[candidate_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != len(reviews):
        raise ValueError(f"Matched {len(matched)} of {len(reviews)} human-reviewed rows")
    if not set(matched["decision"]).issubset({"saved", "rejected"}):
        raise ValueError("Expected human decisions to be saved or rejected")

    human_positive = matched["decision"].eq("saved").to_numpy(dtype=int)
    strict_positive = matched["judge_label"].eq("positive").to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(human_positive, strict_positive).ravel()
    metrics = pd.DataFrame(
        [
            metric_row("audited_precision_ppv", int(tp), int(tp + fp)),
            metric_row("sensitivity", int(tp), int(tp + fn)),
            metric_row("specificity", int(tn), int(tn + fp)),
            metric_row("accuracy", int(tp + tn), len(matched)),
        ]
    )
    scores = matched["judge_keep_score"].to_numpy(dtype=float)
    auc = float(roc_auc_score(human_positive, scores))
    auc_low, auc_high = bootstrap_auc(
        human_positive,
        scores,
        args.bootstrap_draws,
        args.seed,
    )
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {
                        "metric": "judge_keep_score_auroc",
                        "estimate": auc,
                        "ci_low": auc_low,
                        "ci_high": auc_high,
                        "numerator": np.nan,
                        "denominator": len(matched),
                        "interval": "95% row bootstrap",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "human_validation_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        ]
    ).to_csv(args.output_dir / "strict_positive_confusion.csv", index=False)

    audit_rows = matched[["decision", "judge_label", "judge_confidence", "judge_keep_score", "judge_exclusion"]].copy()
    audit_rows.insert(
        0,
        "audit_key",
        [
            hashlib.sha256(f"{conversation_id}\n{text}".encode()).hexdigest()
            for conversation_id, text in zip(
                matched["_conversation_id"],
                matched["_normalized_text"],
                strict=True,
            )
        ],
    )
    audit_rows.to_csv(args.output_dir / "deidentified_audit_rows.csv", index=False)

    hparams = {
        "human_reviews": str(args.human_reviews),
        "candidate_metadata": str(args.candidate_metadata),
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "matched_rows": len(matched),
        "human_positive_rows": int(human_positive.sum()),
        "human_negative_rows": int((1 - human_positive).sum()),
        "selection_note": "Earlier manually reviewed candidate set; not a random sample of the final 433-row public release.",
        "independent_double_coding": False,
        "inter_rater_agreement_available": False,
        "reviewer_assignment_counts_anonymized": sorted(
            (int(value) for value in reviews["username"].value_counts().tolist()),
            reverse=True,
        )
        if "username" in reviews
        else None,
        "strict_release_rule": "judge_label == positive",
    }
    (args.output_dir / "hparams.json").write_text(
        json.dumps(hparams, indent=2),
        encoding="utf-8",
    )

    plotted = metrics[metrics["metric"].isin(["audited_precision_ppv", "sensitivity", "specificity"])].copy()
    labels = {
        "audited_precision_ppv": "Audited precision",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
    }
    plotted["label"] = plotted["metric"].map(labels)
    y = np.arange(len(plotted))[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.errorbar(
        plotted["estimate"],
        y,
        xerr=np.vstack(
            [
                plotted["estimate"] - plotted["ci_low"],
                plotted["ci_high"] - plotted["estimate"],
            ]
        ),
        fmt="o",
        color="#2A6F97",
        capsize=4,
    )
    ax.set_yticks(y, plotted["label"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Proportion (95% Wilson CI)")
    ax.set_title("Strict verifier rule on 108 human-reviewed candidates", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(args.output_dir / "human_validation.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
