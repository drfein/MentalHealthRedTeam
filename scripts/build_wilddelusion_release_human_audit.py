#!/usr/bin/env python3
"""Build a blinded simple-random audit of the final WildDelusion release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/verification/whitened_top3k_verified_gpt54mini.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/wilddelusion_release_human_audit"),
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--stratify-field",
        default=None,
        help="Optional categorical field represented in the sample.",
    )
    parser.add_argument(
        "--min-per-stratum",
        type=int,
        default=0,
        help="Minimum sampled rows per stratum before proportional allocation.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_context(row: dict[str, Any]) -> str:
    messages = row["messages"]
    target_index = int(row["target_message_index"])
    if not 0 <= target_index < len(messages):
        raise ValueError(f"Invalid target index for {row.get('stable_key')}")

    opening: list[int] = []
    opening_counts = {"user": 0, "assistant": 0}
    for index, message in enumerate(messages[: target_index + 1]):
        role = message.get("role")
        if role in opening_counts and opening_counts[role] < 2:
            opening.append(index)
            opening_counts[role] += 1

    preceding: list[int] = []
    preceding_counts = {"user": 0, "assistant": 0}
    for index in range(target_index - 1, -1, -1):
        role = messages[index].get("role")
        if role in preceding_counts and preceding_counts[role] < 5:
            preceding.append(index)
            preceding_counts[role] += 1
        if all(count == 5 for count in preceding_counts.values()):
            break

    selected = sorted(set(opening + preceding + [target_index]))
    rendered = []
    previous = -1
    for index in selected:
        if previous >= 0 and index > previous + 1:
            rendered.append("[... omitted earlier context ...]")
        message = messages[index]
        role = str(message.get("role", "unknown")).upper()
        marker = " [FLAGGED TARGET]" if index == target_index else ""
        content = " ".join(str(message.get("content", "")).split())
        rendered.append(f"{role}{marker}: {content}")
        previous = index
    return "\n\n".join(rendered)


def sample_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    rng: np.random.Generator,
    stratify_field: str | None,
    min_per_stratum: int,
) -> np.ndarray:
    if not stratify_field:
        return rng.choice(len(rows), size=sample_size, replace=False)
    by_stratum: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_stratum.setdefault(str(row.get(stratify_field, "missing")), []).append(index)
    minimum_total = sum(min(min_per_stratum, len(indices)) for indices in by_stratum.values())
    if minimum_total > sample_size:
        raise ValueError("sample-size is too small for the requested per-stratum minimum")

    chosen: list[int] = []
    remaining: list[int] = []
    for stratum in sorted(by_stratum):
        indices = np.array(by_stratum[stratum], dtype=int)
        rng.shuffle(indices)
        minimum = min(min_per_stratum, len(indices))
        chosen.extend(indices[:minimum].tolist())
        remaining.extend(indices[minimum:].tolist())
    extra = sample_size - len(chosen)
    if extra:
        chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return np.array(chosen, dtype=int)


def main() -> None:
    args = parse_args()
    positives = [
        row for row in read_jsonl(args.input) if row.get("judge_label") == "positive"
    ]
    if not 0 < args.sample_size <= len(positives):
        raise ValueError("sample-size must be between 1 and the release size")

    rng = np.random.default_rng(args.seed)
    sampled_indices = sample_indices(
        positives,
        sample_size=args.sample_size,
        rng=rng,
        stratify_field=args.stratify_field,
        min_per_stratum=args.min_per_stratum,
    )
    rng.shuffle(sampled_indices)
    sampled = [positives[int(index)] for index in sampled_indices]

    review_rows = []
    key_rows = []
    for position, (source_index, row) in enumerate(
        zip(sampled_indices, sampled, strict=True), start=1
    ):
        review_id = f"WD-{position:03d}"
        review_rows.append(
            {
                "review_id": review_id,
                "conversation_context": compact_context(row),
                "decision": "",
                "exclusion_reason": "",
                "confidence_1_to_5": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "release_row_index": int(source_index),
                "source": row["source"],
                "split": row["split"],
                "row_offset": row.get("row_offset"),
                "conversation_id": row["conversation_id"],
                "message_hash": row["message_hash"],
                "target_message_index": row["target_message_index"],
                "discovery_split": row.get("discovery_split"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows).to_csv(args.output_dir / "blinded_review.csv", index=False)
    pd.DataFrame(key_rows).to_csv(args.output_dir / "audit_key.csv", index=False)
    source_counts = pd.Series([row["source"] for row in sampled]).value_counts()
    stratum_counts = (
        pd.Series([row.get(args.stratify_field) for row in sampled]).value_counts()
        if args.stratify_field
        else pd.Series(dtype=int)
    )
    manifest = {
        "input": str(args.input),
        "release_rule": "judge_label == positive",
        "release_rows": len(positives),
        "sampling": (
            "stratified random sample without replacement"
            if args.stratify_field
            else "simple random sample without replacement"
        ),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "stratify_field": args.stratify_field,
        "min_per_stratum": args.min_per_stratum,
        "sample_stratum_counts": {
            str(stratum): int(count) for stratum, count in stratum_counts.items()
        },
        "sample_source_counts": {
            str(source): int(count) for source, count in source_counts.items()
        },
        "blinding": [
            "source and provenance",
            "retrieval score",
            "message-judge score and rationale",
            "conversation-verifier output",
        ],
        "context_policy": "first 2 user and assistant messages plus up to 5 user and assistant messages preceding the flagged turn",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
