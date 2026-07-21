#!/usr/bin/env python3
"""Analyze locked human reviews of the final WildDelusion release audit."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


DECISIONS = {"positive", "negative", "uncertain"}
EXCLUSIONS = {
    "none",
    "roleplay",
    "fiction_or_story",
    "joke_or_absurd",
    "translation_or_text_task",
    "third_party_or_quoted",
    "ordinary_plausible",
    "insufficient_context",
    "other",
}


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def load_review(path: Path, rater: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"review_id", "decision", "exclusion_reason", "confidence_1_to_5"}
    if missing := required - set(frame):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    decisions = frame["decision"].str.strip().str.lower()
    if not set(decisions).issubset(DECISIONS) or (decisions == "").any():
        invalid = sorted(set(decisions) - DECISIONS)
        raise ValueError(f"{path} has incomplete or invalid decisions: {invalid}")
    exclusions = frame["exclusion_reason"].str.strip().str.lower()
    if not set(exclusions).issubset(EXCLUSIONS) or (exclusions == "").any():
        invalid = sorted(set(exclusions) - EXCLUSIONS)
        raise ValueError(f"{path} has incomplete or invalid exclusions: {invalid}")
    if ((decisions == "positive") & (exclusions != "none")).any():
        raise ValueError(f"{path}: positive decisions must use exclusion_reason=none")
    if ((decisions != "positive") & (exclusions == "none")).any():
        raise ValueError(f"{path}: non-positive decisions require an exclusion reason")
    confidence = pd.to_numeric(frame["confidence_1_to_5"], errors="coerce")
    if confidence.isna().any() or not confidence.between(1, 5).all():
        raise ValueError(f"{path}: confidence must be complete and between 1 and 5")
    if frame["review_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate review IDs")
    return pd.DataFrame(
        {
            "review_id": frame["review_id"].astype(str),
            f"decision__{rater}": decisions,
            f"exclusion__{rater}": exclusions,
            f"confidence__{rater}": confidence.astype(int),
        }
    )


def metric_row(rater: str, metric: str, numerator: int, denominator: int) -> dict[str, Any]:
    low, high = wilson(numerator, denominator)
    return {
        "rater": rater,
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator / denominator,
        "ci_low": low,
        "ci_high": high,
        "interval": "95% Wilson",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--rater-names", nargs="+", default=None)
    parser.add_argument("--audit-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    names = args.rater_names or [f"rater_{i + 1}" for i in range(len(args.reviews))]
    if len(names) != len(args.reviews) or len(set(names)) != len(names):
        parser.error("Provide one unique rater name per review file")

    key = pd.read_csv(args.audit_key)
    if key["review_id"].duplicated().any():
        raise ValueError("Audit key contains duplicate review IDs")
    frame = key.copy()
    expected_ids = set(frame["review_id"].astype(str))
    for path, name in zip(args.reviews, names, strict=True):
        review = load_review(path, name)
        if set(review["review_id"]) != expected_ids:
            raise ValueError(f"{path} review IDs do not exactly match the audit key")
        frame = frame.merge(review, on="review_id", validate="one_to_one")

    metrics: list[dict[str, Any]] = []
    for name in names:
        decisions = frame[f"decision__{name}"]
        positive = int(decisions.eq("positive").sum())
        negative = int(decisions.eq("negative").sum())
        uncertain = int(decisions.eq("uncertain").sum())
        metrics.append(metric_row(name, "strict_release_precision", positive, len(frame)))
        metrics.append(
            metric_row(name, "resolved_case_precision", positive, positive + negative)
        )
        metrics.append(metric_row(name, "uncertain_rate", uncertain, len(frame)))

    agreements = []
    for first, second in combinations(names, 2):
        a = frame[f"decision__{first}"]
        b = frame[f"decision__{second}"]
        agreements.append(
            {
                "first_rater": first,
                "second_rater": second,
                "n": len(frame),
                "three_class_raw_agreement": float(a.eq(b).mean()),
                "three_class_cohen_kappa": float(cohen_kappa_score(a, b)),
                "positive_vs_other_raw_agreement": float(
                    a.eq("positive").eq(b.eq("positive")).mean()
                ),
                "positive_vs_other_cohen_kappa": float(
                    cohen_kappa_score(a.eq("positive"), b.eq("positive"))
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.output_dir / "precision_metrics.csv", index=False)
    pd.DataFrame(agreements).to_csv(args.output_dir / "rater_agreement.csv", index=False)
    frame.to_csv(args.output_dir / "audit_analysis_rows.csv", index=False)
    (args.output_dir / "hparams.json").write_text(
        json.dumps(
            {
                "reviews": [str(path) for path in args.reviews],
                "rater_names": names,
                "audit_key": str(args.audit_key),
                "sample_size": len(frame),
                "strict_release_precision": "positive / all audited rows; uncertain counts as non-positive",
                "resolved_case_precision": "positive / (positive + negative); uncertain excluded",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
