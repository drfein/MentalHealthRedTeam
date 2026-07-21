#!/usr/bin/env python3
"""Build a blinded third-review sheet for disagreements between locked raters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RESPONSE_SCORE_COLUMNS = (
    "human_reality_endorsement_score_0_to_5",
    "human_frame_adherence_score_0_to_2",
)
RELEASE_COLUMNS = ("decision", "exclusion_reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("release", "response"), required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--blinded-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def normalized_review(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"review_id", *columns}
    if missing := required - set(frame):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["review_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate review IDs")
    if frame[list(columns)].eq("").any().any():
        raise ValueError(f"{path} contains incomplete adjudication fields")
    result = frame[["review_id", *columns]].copy()
    result["review_id"] = result["review_id"].astype(str)
    return result.set_index("review_id")


def main() -> None:
    args = parse_args()
    if len(args.reviews) < 2:
        raise ValueError("At least two independent review files are required")
    columns = RESPONSE_SCORE_COLUMNS if args.kind == "response" else RELEASE_COLUMNS
    blinded = pd.read_csv(args.blinded_review, keep_default_na=False)
    if "review_id" not in blinded or blinded["review_id"].duplicated().any():
        raise ValueError("Blinded review must contain unique review IDs")
    blinded["review_id"] = blinded["review_id"].astype(str)
    expected_ids = set(blinded["review_id"])

    reviews = [normalized_review(path, columns) for path in args.reviews]
    for path, review in zip(args.reviews, reviews, strict=True):
        if set(review.index) != expected_ids:
            raise ValueError(f"{path} review IDs do not match the blinded sample")

    disagreement = pd.Series(False, index=blinded["review_id"])
    for column in columns:
        values = pd.concat(
            [review[column].rename(str(index)) for index, review in enumerate(reviews)],
            axis=1,
        )
        disagreement |= values.nunique(axis=1).gt(1)
    disagreement_ids = set(disagreement[disagreement].index)
    output = blinded[blinded["review_id"].isin(disagreement_ids)].copy()
    for column in output.columns:
        if column.startswith("human_") or column in {
            "decision",
            "exclusion_reason",
            "confidence_1_to_5",
            "notes",
        }:
            output[column] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "blinded_adjudication.csv"
    output.to_csv(output_path, index=False)
    manifest = {
        "kind": args.kind,
        "seed": args.seed,
        "source_reviews": [str(path) for path in args.reviews],
        "blinded_review": str(args.blinded_review),
        "total_review_rows": len(output),
        "disagreement_fields": list(columns),
        "blinding": "third reviewer sees content and row ID, but not prior decisions or judge metadata",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
